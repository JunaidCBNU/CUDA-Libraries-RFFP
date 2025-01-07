#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>  
#include <cuda_bf16.h>
#include <cfloat>

struct DecomposedFloat {
    int sign;
    int exponent;
    int mantissa;
};
__device__ __forceinline__ int getCustomBias(int exp_bits)
{
    // Example: If exp_bits == 4, use 12; otherwise, fallback to standard formula.
    if (exp_bits == 4) {
        return 10;
    } else {
        // Default formula (symmetric bias) = 2^(exp_bits - 1) - 1
        return (1 << (exp_bits - 1)) - 1;
    }
}
// Convert a float to a custom floating-point representation (RFFP format)
__device__ DecomposedFloat Floating2Binary_RFFP(float* num_ptr, int Exponent_Bit, int Mantissa_Bit) {
    float num = *num_ptr;
    DecomposedFloat result;
    unsigned int num_bits = *(reinterpret_cast<unsigned int*>(&num));

    // Extract sign, exponent, and mantissa
    result.sign = (num_bits >> 31) & 0x1;
    unsigned int raw_exponent = (num_bits >> 23) & 0xFF;
    unsigned int raw_mantissa = num_bits & 0x7FFFFF;

    // Calculate bias and truncation for mantissa
    int bias = getCustomBias(Exponent_Bit);
    int mantissa_shift = 23 - Mantissa_Bit;
    unsigned int mantissa_truncated = raw_mantissa >> mantissa_shift;
    unsigned int round_bit = (raw_mantissa >> (mantissa_shift - 1)) & 1;
    unsigned int sticky_bits = raw_mantissa & ((1 << (mantissa_shift - 1)) - 1);

    // Apply rounding rules
    bool round_up = round_bit && (sticky_bits != 0 || (mantissa_truncated & 1));
    if (round_up) {
        mantissa_truncated++;
        if (mantissa_truncated >= (1 << Mantissa_Bit)) {
            mantissa_truncated = 0;
            raw_exponent++;
        }
    }

    // Handle special cases (NaN and Infinity)
    unsigned int max_exponent = (1 << Exponent_Bit) - 1;
    if (raw_exponent == 0xFF) {
        result.exponent = max_exponent;
        result.mantissa = (raw_mantissa != 0) ? 1 : 0;
        return result;
    }

    // Adjust exponent and handle overflows/underflows
    int adjusted_exponent = static_cast<int>(raw_exponent) - 127 + bias;
    result.exponent = adjusted_exponent;
    result.mantissa = mantissa_truncated;

    if (adjusted_exponent < 0) {
        result.exponent = 0;
        result.mantissa = raw_mantissa >> (1 - adjusted_exponent);
    } else if (adjusted_exponent >= max_exponent) {
        result.exponent = max_exponent;
        result.mantissa = 0;
    }

    return result;
}

__device__ float Converter_to_FP(int sign_c, int exponent_c, int mantissa_c, int exp_bits, int mantissa_bits) {
    int enhanced_exp_offset = (1 << (exp_bits + 1))-1; 

    // Handle special cases (Infinity and Zero)
    if (exponent_c == 0xFF) {
        return sign_c ? -INFINITY : INFINITY;
    } else if (exponent_c == 0) {
        return sign_c ? -0.0f : 0.0f;
    }

    // Normalize mantissa by keeping only e.g. 18 bits if accumulation is done in 19-bits
    int mantissa_c_int = mantissa_c & 0x3FFFF;  // Removing the implicit 1, e.g. 19th Bit)

    // Adjust exponent for custom precision
    int exponent_mul_shift = exponent_c;
    if (exponent_c != 0 && mantissa_c != 0) {
        exponent_mul_shift += (18 - mantissa_bits*2);
    }

    // Convert mantissa to floating point using 8 bits instead of 23
    float mantissa = (1.0f + mantissa_c_int * powf(2.0f, -18));  // Changed from -23 to -8
    float result = (sign_c == 0 ? 1.0f : -1.0f) * ldexpf(mantissa, exponent_mul_shift - enhanced_exp_offset);

    return result;
}


// Optimized Multiply Function
__device__ void multiply(int index_a, int index_b, DecomposedFloat converted_a, DecomposedFloat converted_b,  
                         int exp_bits, int mantissa_bits, 
                         int& mantissa_mul, int& sign_mul, int& exp_mul) { 
 
    int mantissa_mask = (1 << mantissa_bits) - 1;  // Mask to extract mantissa_bits 
    int implicit_bit = (1 << mantissa_bits); 
    int mantissa_explicit_a = (converted_a.exponent != 0 ? implicit_bit : 0) | (converted_a.mantissa & mantissa_mask); 
    int mantissa_explicit_b = (converted_b.exponent != 0 ? implicit_bit : 0) | (converted_b.mantissa & mantissa_mask); 
 
    int exponent_offset = getCustomBias(exp_bits);
    int min_exp = 0;
    int max_exp = (1 << (exp_bits + 2))-1; // If 4B exponent is used then we increase the exponent to 6B
    int enhanced_exp_offset = (1 << (exp_bits + 1))-1;  // Enhanced exponent offset for 6-bit exponent bias
 
    // Handle special cases (NaN, Infinity, Zero)
    if ((converted_a.exponent == max_exp && converted_a.mantissa != 0) || (converted_b.exponent == max_exp && converted_b.mantissa != 0)) { 
        exp_mul = max_exp; 
        mantissa_mul = 1;  // Non-zero mantissa for NaN 
        sign_mul = 0;  // Sign bit is ignored for NaN 
        return; 
    } 
    if ((converted_a.exponent == max_exp && converted_a.mantissa == 0) || (converted_b.exponent == max_exp && converted_b.mantissa == 0)) { 
        exp_mul = max_exp; 
        mantissa_mul = 0; 
        sign_mul = converted_a.sign ^ converted_b.sign; 
        return; 
    } 
    if (converted_a.exponent == 0 || converted_b.exponent == 0) { 
        exp_mul = 0; 
        mantissa_mul = 0; 
        sign_mul = 0; 
        return; 
    } 

    // Regular multiplication for non-special cases
    sign_mul = converted_a.sign ^ converted_b.sign; 
    mantissa_mul = mantissa_explicit_a * mantissa_explicit_b; 
    exp_mul = converted_a.exponent + converted_b.exponent - 2*exponent_offset + enhanced_exp_offset; // 6-Bits Exponent Bias is added
 
    // Handle overflow and underflow
    if (exp_mul > max_exp) { 
        exp_mul = max_exp;  // Set to infinity 
        mantissa_mul = 0; 
    } else if (exp_mul < min_exp) { 
        exp_mul = 0;  // Handle underflow 
        mantissa_mul = 0; 
    } 
}
// Optimized Accumulate Function
__device__ void accumulate(int& exp_sum, int& mantissa_sum, int& sign_sum,    
                           int exp_mul, int mantissa_mul, int sign_mul) {                        

    int exponent_diff;   
    if (exp_sum != 0 && exp_mul != 0) {   
        exponent_diff = exp_sum - exp_mul;   
    } else if (exp_sum == 0) {   
        exponent_diff = -exp_mul;   
    } else {   
        exponent_diff = exp_sum;   
    }   
   
    // Align mantissas based on the exponent difference   
    int temp_mantissa_a = mantissa_sum;   
    int temp_mantissa_b = mantissa_mul;   
    // printf("Mantissa A: %d, Mantissa B: %d\n", temp_mantissa_a, temp_mantissa_b);
    if (exponent_diff > 0) {   
        temp_mantissa_b >>= exponent_diff;   
    } else if (exponent_diff < 0) {   
        temp_mantissa_a >>= -exponent_diff;   
        exp_sum = exp_mul;  // Update exponent to the larger one   
    }   
   
    // Add or subtract mantissas based on sign   
    if (sign_sum == sign_mul) {   
        mantissa_sum = temp_mantissa_a + temp_mantissa_b;   
    } else {   
        if (temp_mantissa_a >= temp_mantissa_b) {   
            mantissa_sum = temp_mantissa_a - temp_mantissa_b;   
        } else {   
            sign_sum = sign_mul;   
            mantissa_sum = temp_mantissa_b - temp_mantissa_a;   
        }   
    }   
  
    // Normalize to ensure mantissa is 19 bits with MSB as 1  
    const int target_mantissa_bits = 19;  // Changed from 24 to 9
    if (__clz(mantissa_sum) < 32 - target_mantissa_bits) {  
        int shift_amount = __clz(mantissa_sum) - (32 - target_mantissa_bits);   
        mantissa_sum >>= shift_amount;   
        exp_sum += shift_amount;        // Adjust exponent accordingly  
    }  
    // Shift left to ensure the MSB is set to 1 
    else {  
        while ((__clz(mantissa_sum) > (32 - target_mantissa_bits)) && mantissa_sum != 0) {  
            mantissa_sum <<= 1;  
            exp_sum -= 1;  // Adjust exponent to maintain the numeric value  
        }  
    }  

    // Add masking to ensure mantissa doesn't exceed 9 bits
    mantissa_sum &= ((1 << target_mantissa_bits) - 1);
}

__device__ void add_bias(int& exp_sum, int& mantissa_sum, int& sign_sum,    
                           int exp_mul, int mantissa_bias, int sign_mul, int mantissa_bits) {  
    int mantissa_mask = (1 << mantissa_bits) - 1;  // Mask to extract mantissa_bits 
    int implicit_bit = (1 << mantissa_bits); 
    int mantissa_mul = 0;
    mantissa_mul = (exp_mul != 0 ? implicit_bit : 0) | (mantissa_bias & mantissa_mask);  
    
    int exponent_diff;  
    if (exp_mul != 0) 
        exp_mul = exp_mul - 3;  // Adjust exponent to 6-bit bias
    else
        exp_mul = 0;  

    if (exp_sum != 0 && exp_mul != 0) {   
        exponent_diff = exp_sum - exp_mul;   
    } else if (exp_sum == 0) {   
        exponent_diff = -exp_mul;   
    } else {   
        exponent_diff = exp_sum;   
    }   
   
    // Align mantissas based on the exponent difference   
    int temp_mantissa_a = mantissa_sum;   
    int temp_mantissa_b = mantissa_mul;   
    if (exponent_diff > 0) {   
        temp_mantissa_b >>= exponent_diff;   
    } else if (exponent_diff < 0) {   
        temp_mantissa_a >>= -exponent_diff;   
        exp_sum = exp_mul;  // Update exponent to the larger one   
    }   
   
    // Add or subtract mantissas based on sign   
    if (sign_sum == sign_mul) {   
        mantissa_sum = temp_mantissa_a + temp_mantissa_b;   
    } else {   
        if (temp_mantissa_a >= temp_mantissa_b) {   
            mantissa_sum = temp_mantissa_a - temp_mantissa_b;   
        } else {   
            sign_sum = sign_mul;   
            mantissa_sum = temp_mantissa_b - temp_mantissa_a;   
        }   
    }   

    // Normalize to ensure mantissa is 9 bits with MSB as 1  
    const int target_mantissa_bits = 19;  // Changed from 24 to 9
    if (__clz(mantissa_sum) < 32 - target_mantissa_bits) {  
        int shift_amount = __clz(mantissa_sum) - (32 - target_mantissa_bits);   
        mantissa_sum >>= shift_amount;   
        exp_sum += shift_amount;        // Adjust exponent accordingly  
    }  
    // Shift left to ensure the MSB is set to 1 
    else {  
        while ((__clz(mantissa_sum) > (32 - target_mantissa_bits)) && mantissa_sum != 0) {  
            mantissa_sum <<= 1;  
            exp_sum -= 1;  // Adjust exponent to maintain the numeric value  
        }  
    }  

    // Add masking to ensure mantissa doesn't exceed 9 bits
    // mantissa_sum &= ((1 << target_mantissa_bits) - 1);
}

__global__ void convolutionKernel_WB(int N, int C, int H, int W, 
    float* input, int F, int HH, int WW, 
    float* kernel, float* bias, float* output, int H_out, int W_out, int pad, int stride, int exp_bits) { 
 
    int w_out = blockIdx.x * blockDim.x + threadIdx.x; 
    int h_out = blockIdx.y * blockDim.y + threadIdx.y; 
    int f = blockIdx.z; 

    int mantissa_bits = 3;

    // printf("Generalized mantissa width after accumulation: %d bits\n", m_bits);
    if (w_out < W_out && h_out < H_out && f < F) { 

        for (int n = 0; n < N; n++) { 
            int exp_sum = 0; 
            int mantissa_sum = 0; 
            int sign_sum = 0; 
 
            for (int c = 0; c < C; c++) {  
                for (int hh = 0; hh < HH; hh++) {  
                    for (int ww = 0; ww < WW; ww++) {  
                        int h_in = h_out * stride + hh - pad;  
                        int w_in = w_out * stride + ww - pad;  
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {  
                            // Calculate input and weight indices
                            int index_a = n * (C * H * W) + c * (H * W) + h_in * W + w_in;  
                            int index_b = f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww;  

                            // Use Floating2Binary_RFFP to convert directly from 23 to custom mantissa bits
                            DecomposedFloat FMAP = Floating2Binary_RFFP(&input[index_a], exp_bits, mantissa_bits);  
                            DecomposedFloat WEIGHT = Floating2Binary_RFFP(&kernel[index_b], exp_bits, mantissa_bits);  

                            // No need for intermediate RFFP_CONVERTER call, as Floating2Binary_RFFP already applies custom bit logic
                            DecomposedFloat converted_FMAP = FMAP;  
                            DecomposedFloat converted_WEIGHT = WEIGHT;  

                            // Initialize multiplication and accumulation variables  
                            int mantissa_mul, sign_mul, exp_mul;  

                            // Perform multiplication with directly converted values
                            multiply(index_a, index_b, converted_FMAP, converted_WEIGHT, exp_bits, mantissa_bits, mantissa_mul, sign_mul, exp_mul);  

                            // Accumulate the multiplication results
                            accumulate(exp_sum, mantissa_sum, sign_sum, exp_mul, mantissa_mul, sign_mul);  
                        }  
                    }  
                }  
            }  

            // Convert bias to lower precision
            // Add bias with correct precision
            DecomposedFloat BIAS = Floating2Binary_RFFP(&bias[f], exp_bits+2, 9);  // 6-bit exponent, 16-bit mantissa
            add_bias(exp_sum, mantissa_sum, sign_sum, BIAS.exponent, BIAS.mantissa, BIAS.sign, 9);

            // Convert the accumulated convolution result to FP32
            float convertedValue;
            convertedValue = Converter_to_FP(sign_sum, exp_sum, mantissa_sum, exp_bits, mantissa_bits);

            // Store the result in the output tensor
            int index_out = n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out;
            output[index_out] = convertedValue;   
        } 
    } 
}

extern "C" {  
    void conv2d_WB(int N, int C, int H, int W,  
        float* input, int F, int HH, int WW,  
        float* kernel, float* bias, float* output, int pad, int stride, int exp_bits) {  

        int H_out = 1 + (H + 2 * pad - HH) / stride;  
        int W_out = 1 + (W + 2 * pad - WW) / stride;  

        // Define grid and block dimensions for CUDA threads  
        dim3 blockDim(16, 16, 1);  
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, (H_out + blockDim.y - 1) / blockDim.y, F);  

        // Launch CUDA kernel for convolution  
        convolutionKernel_WB<<<gridDim, blockDim>>>(N, C, H, W, input, F, HH, WW, kernel, bias, output, H_out, W_out, pad, stride, exp_bits);

        // Ensure kernel execution is complete
        cudaDeviceSynchronize();

        // Check for CUDA errors  
        cudaError_t err = cudaGetLastError();  
        if (err != cudaSuccess) {  
            printf("CUDA Error: %s\n", cudaGetErrorString(err));  
        }  
    }  
}  


__global__ void depthwiseConvolutionKernel(int N, int C, int H, int W, 
    float* input, int HH, int WW,  
    float* kernel, float* bias, float* output, int H_out, int W_out, int pad, int stride, int exp_bits) {

    // Get thread index for spatial location and channel
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;  // Output x coordinate
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;  // Output y coordinate
    int c = blockIdx.z;  // Channel index (depthwise convolution processes each channel independently)
    
    int mantissa_bits = 3;
    
    if (w_out < W_out && h_out < H_out && c < C) {
        for (int n = 0; n < N; n++) {  // Batch dimension
            
            int exp_sum = 0; 
            int mantissa_sum = 0; 
            int sign_sum = 0; 
            float convertedValue = 0;

            // Perform the convolution for the current channel
            for (int hh = 0; hh < HH; hh++) {
                for (int ww = 0; ww < WW; ww++) {
                    int h_in = h_out * stride + hh - pad;  // Input y coordinate
                    int w_in = w_out * stride + ww - pad;  // Input x coordinate

                    // Ensure input coordinates are within valid bounds
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int index_a = n * (C * H * W) + c * (H * W) + h_in * W + w_in; 
                        int index_b = c * (HH * WW) + hh * WW + ww; 
                        // Convert input and kernel values to Custom Format
                        DecomposedFloat FMAP = Floating2Binary_RFFP(&input[index_a], exp_bits, mantissa_bits); 
                        DecomposedFloat WEIGHT = Floating2Binary_RFFP(&kernel[index_b], exp_bits, mantissa_bits); 

                        DecomposedFloat converted_FMAP =   FMAP; 
                        DecomposedFloat converted_WEIGHT = WEIGHT; 

                        // Initialize multiplication and accumulation variables 
                        int mantissa_mul, sign_mul, exp_mul; 

                        // Call multiply function with converted values 
                        multiply(index_a, index_b, converted_FMAP, converted_WEIGHT, exp_bits, mantissa_bits, mantissa_mul, sign_mul, exp_mul); 

                        // Accumulate results 
                        accumulate(exp_sum, mantissa_sum, sign_sum, exp_mul, mantissa_mul, sign_mul);
                    }
                }
            }

            // Convert bias to lower precision

            DecomposedFloat BIAS = Floating2Binary_RFFP(&bias[c], exp_bits+2, 9);  // 4+2-bit exponent, 9-bit mantissa
            add_bias(exp_sum, mantissa_sum, sign_sum, BIAS.exponent, BIAS.mantissa, BIAS.sign, 9);            

            // Convert the accumulated convolution result to FP32
            convertedValue = Converter_to_FP(sign_sum, exp_sum, mantissa_sum, exp_bits, mantissa_bits);

            int index_out =n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
            // Convert the result back to float and store it in the output tensor
            output[index_out] = convertedValue;
        }
    }
}

extern "C" {
    void depthwise_conv2d(int N, int C, int H, int W,
        float* input, int HH, int WW,
        float* kernel, float* bias, float* output, int pad, int stride, int exp_bits) {

        // Calculate output dimensions
        int H_out = 1 + (H + 2 * pad - HH) / stride;
        int W_out = 1 + (W + 2 * pad - WW) / stride;

        // Ensure valid output dimensions
        if (H_out <= 0 || W_out <= 0) {
            printf("Error: Invalid output dimensions.\n");
            return;
        }

        // Define block dimensions (16x16 threads per block)
        dim3 blockDim(16, 16, 1);  

        // Calculate grid dimensions based on the output dimensions
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, 
                     (H_out + blockDim.y - 1) / blockDim.y, 
                     C);  // One block per channel

        // Launch CUDA kernel for depthwise convolution
        depthwiseConvolutionKernel<<<gridDim, blockDim>>>(N, C, H, W, input, HH, WW, kernel, bias, output, H_out, W_out, pad, stride, exp_bits);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        // Synchronize the device
        cudaDeviceSynchronize();
    }
}

__global__ void convolutionKernel_16(int N, int C, int H, int W,
    float* input, int F, int HH, int WW,
    float* kernel, float* bias, float* output, int H_out, int W_out, int pad, int stride) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (w_out < W_out && h_out < H_out && f < F) {
        for (int n = 0; n < N; n++) {
            __nv_bfloat16 sum_bf16 = __float2bfloat16(0.0f);  // Initialize sum as bfloat16

            for (int c = 0; c < C; c++) {
                for (int hh = 0; hh < HH; hh++) {
                    for (int ww = 0; ww < WW; ww++) {
                        int h_in = h_out * stride + hh - pad;
                        int w_in = w_out * stride + ww - pad;

                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            // Convert float inputs to bfloat16
                            __nv_bfloat16 input_bf16 = __float2bfloat16(input[n * (C * H * W) + c * (H * W) + h_in * W + w_in]);
                            __nv_bfloat16 kernel_bf16 = __float2bfloat16(kernel[f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww]);

                            // Perform multiplication and addition in bfloat16
                            sum_bf16 = __hadd(sum_bf16, __hmul(input_bf16, kernel_bf16));
                        }
                    }
                }
            }

            // Convert bias to bfloat16 and add
            __nv_bfloat16 bias_bf16 = __float2bfloat16(bias[f]);
            sum_bf16 = __hadd(sum_bf16, bias_bf16);

            // Convert the result back to float and store in output
            output[n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out] += __bfloat162float(sum_bf16);
        }
    }
}
extern "C" {
    void conv2d_16(int N, int C, int H, int W,
        float* input, int F, int HH, int WW,
        float* kernel, float* bias, float* output, int pad, int stride) {

        int H_out = 1 + (H + 2 * pad - HH) / stride;
        int W_out = 1 + (W + 2 * pad - WW) / stride;

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim(16, 16, 1);
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, (H_out + blockDim.y - 1) / blockDim.y, F);

        // Launch CUDA kernel for convolution
        convolutionKernel_16<<<gridDim, blockDim>>>(N, C, H, W, input, F, HH, WW, kernel, bias, output, H_out, W_out, pad, stride);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
    }
}
// Function for CPU-based convolution with bias
void cpu_convolution_with_bias(int N, int C, int H, int W, float* input, int F, int HH, int WW, float* kernel, float* bias, float* output, int pad, int stride) {
    int H_out = 1 + (H + 2 * pad - HH) / stride;
    int W_out = 1 + (W + 2 * pad - WW) / stride;

    // Initialize the output to zero
    for (int i = 0; i < N * F * H_out * W_out; i++) {
        output[i] = 0.0f;
    }

    // Perform convolution
    for (int n = 0; n < N; n++) {  // For each batch element
        for (int f = 0; f < F; f++) {  // For each filter
            for (int h_out = 0; h_out < H_out; h_out++) {  // Output height
                for (int w_out = 0; w_out < W_out; w_out++) {  // Output width
                    float sum = 0.0f;
                    for (int c = 0; c < C; c++) {  // For each input channel
                        for (int hh = 0; hh < HH; hh++) {  // Kernel height
                            for (int ww = 0; ww < WW; ww++) {  // Kernel width
                                int h_in = h_out * stride + hh - pad;
                                int w_in = w_out * stride + ww - pad;
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    sum += input[n * (C * H * W) + c * (H * W) + h_in * W + w_in] *
                                           kernel[f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww];
                                }
                            }
                        }
                    }
                    // Add bias
                    sum += bias[f];
                    // Store result
                    output[n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out] = sum;
                    
                }
            }
        }
    }
}

// Define the function signature for the convolution kernel
extern "C" void conv2d_WB(int N, int C, int H, int W, float* input, int F, int HH, int WW, float* kernel, float* bias, float* output, int pad, int stride, int exp_bits);
extern "C" void conv2d_16(int N, int C, int H, int W, float* input, int F, int HH, int WW, float* kernel, float* bias, float* output, int pad, int stride);

int main() {
    // Example parameters
    int N = 1;        // Batch size
    int C = 100;        // Input channels
    int H = 3;        // Input height
    int W = 3;        // Input width
    int F = 1;        // Number of filters (output channels)
    int HH = 3;       // Kernel height
    int WW = 3;       // Kernel width
    int pad = 1;      // Padding
    int stride = 1;   // Stride
    int exp_bits = 5; // Exponent bits for FP

    // Calculate output dimensions
    int H_out = 1 + (H + 2 * pad - HH) / stride;
    int W_out = 1 + (W + 2 * pad - WW) / stride;

    // Allocate memory on the host
    float h_input[N * C * H * W];
    float h_kernel[F * C * HH * WW];
    float h_bias[F];
    float h_output_CUSTOM[N * F * H_out * W_out];
    float h_output_16[N * F * H_out * W_out];
    float h_output_cpu[N * F * H_out * W_out];
    

    // Initialize input, kernel, and bias with some values
    // Initialize input, kernel, and bias with some values
    for (int i = 0; i < N * C * H * W; i++) {
        h_input[i] = (float)(0.0123456789);  // Input: sequential numbers
        //debug_conversion(h_input[i]);
        //debug_bfloat(h_input[i]);
    }
    for (int i = 0; i < F * C * HH * WW; i++) {
        h_kernel[i] = (float)(0.0123456789); // Kernel: sequential numbers
    }
    for (int i = 0; i < F; i++) {
        h_bias[i] = (float)( 0.524);   // Bias: small values
    }


    // Allocate memory on the device
    float *d_input, *d_kernel, *d_bias, *d_output;
    float *d_input16, *d_kernel16, *d_bias16, *d_output16;
    cudaMalloc(&d_input, N * C * H * W * sizeof(float));
    cudaMalloc(&d_kernel, F * C * HH * WW * sizeof(float));
    cudaMalloc(&d_bias, F * sizeof(float));
    cudaMalloc(&d_output, N * F * H_out * W_out * sizeof(float));
    
    cudaMalloc(&d_input16, N * C * H * W * sizeof(float));
    cudaMalloc(&d_kernel16, F * C * HH * WW * sizeof(float));
    cudaMalloc(&d_bias16, F * sizeof(float));
    cudaMalloc(&d_output16, N * F * H_out * W_out * sizeof(float));

    cudaMemcpy(d_input16, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel16, h_kernel, F * C * HH * WW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias16, h_bias, F * sizeof(float), cudaMemcpyHostToDevice);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, F * C * HH * WW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, F * sizeof(float), cudaMemcpyHostToDevice);



    // Run the custom convolution kernel on the GPU
    //conv2d_WB(N, C, H, W, d_input, F, HH, WW, d_kernel, d_bias, d_output, pad, stride, exp_bits);

    // Copy the GPU result back to the host
    // Run the custom convolution kernel on the GPU
    conv2d_16(N, C, H, W, d_input16, F, HH, WW, d_kernel16, d_bias16, d_output16, pad, stride);
    
    // Run the custom convolution kernel on the GPU
    conv2d_WB(N, C, H, W, d_input, F, HH, WW, d_kernel, d_bias, d_output, pad, stride, exp_bits);

    cudaMemcpy(h_output_CUSTOM, d_output, N * F * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Copy the GPU result back to the host

    cudaMemcpy(h_output_16, d_output16, N * F * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Run the CPU-based convolution for comparison
    cpu_convolution_with_bias(N, C, H, W, h_input, F, HH, WW, h_kernel, h_bias, h_output_cpu, pad, stride);
    

    //Compare the results
    printf("\nComparison (CUSTOM vs CPU):\n");
    for (int i = 0; i < N * F * H_out * W_out; i++) {
        if (fabs(h_output_CUSTOM[i] - h_output_cpu[i]) > 1e-5) {
            printf(" CUSTOM CUDA = %.6f, CPU FP32 = %.6f\n", h_output_CUSTOM[i], h_output_cpu[i]);
        } else {
            printf("Match at index %d: %f\n", i, h_output_CUSTOM[i]);
        }
    }
    // //Compare the results
    // printf("\nComparison (CUSTOM vs GPU vs CPU):\n");
    // for (int i = 0; i < N * F * H_out * W_out; i++) {
    //     if (fabs(h_output_CUSTOM[i] - h_output_cpu[i]) > 1e-5) {
    //         printf(" CUSTOM CUDA = %.6f, GPU Bfloat16 = %.6f, CPU FP32 = %.6f\n", h_output_CUSTOM[i], h_output_16[i], h_output_cpu[i]);
    //     } else {
    //         printf("Match at index %d: %f\n", i, h_output_CUSTOM[i]);
    //     }
    // }
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_bias);
    cudaFree(d_output);

    cudaFree(d_input16);
    cudaFree(d_kernel16);
    cudaFree(d_bias16);
    cudaFree(d_output16);

    return 0;
}