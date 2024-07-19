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
__device__ DecomposedFloat Floating2Binary_RFFP(float* num_ptr, int Exponent_Bit, int Mantissa_Bit) {
    float num = *num_ptr;
    DecomposedFloat result;
    unsigned int num_bits = *(reinterpret_cast<unsigned int*>(&num));

    result.sign = (num_bits >> 31) & 0x1;
    int raw_exponent = (num_bits >> 23) & 0xFF;
    int raw_mantissa = num_bits & 0x7FFFFF;

    int exponent_bias = (1 << (Exponent_Bit - 1)) - 1; 

    // Improved rounding logic for reducing mantissa from 23 to custom bits
    int shift_amount = 23 - Mantissa_Bit;
    int rounding_bit = 1 << (shift_amount - 1);
    int rounding_mask = rounding_bit - 1;
    int mantissa_rounded = (raw_mantissa >> shift_amount) + ((raw_mantissa & rounding_mask) > rounding_bit || ((raw_mantissa & rounding_mask) == rounding_bit && (raw_mantissa >> shift_amount) & 0x1));

    if (mantissa_rounded >= (1 << Mantissa_Bit)) {
        mantissa_rounded = 0; 
        raw_exponent++;
    }

    if (raw_exponent > 0xFE) {
        result.exponent = (1 << Exponent_Bit) - 1;
        result.mantissa = (raw_exponent == 0xFF && raw_mantissa != 0) ? 1 : 0; 
        return result;
    }

    result.exponent = (raw_exponent == 0) ? 0 : raw_exponent + exponent_bias - 127;
    result.mantissa = mantissa_rounded & ((1 << Mantissa_Bit) - 1);

    return result;
}

__device__ DecomposedFloat RFFP_CONVERTER(int* sign, int* exponent, int* mantissa, int exp_bits, int mantissa_bits, int exp_offset) {

    DecomposedFloat result;
    
    result.sign = *sign;
    int exponent_val = *exponent;
    int mantissa_val = *mantissa;
    // int condition_mantissa = exponent_val != 0 && mantissa_val != 0;
    int condition_exponent = exponent_val != 0;
    int shifted_exponent;

    if (exp_bits == 0) {
        shifted_exponent = condition_exponent ? exponent_val - exp_offset : 0;
    } else {
        shifted_exponent = condition_exponent ? exponent_val - (128 - (1 << (exp_bits - 1))) : 0;
    }
    result.exponent = condition_exponent ? shifted_exponent + (7 - mantissa_bits) : 0;

    // int mantissa_explicit = condition_mantissa ? (mantissa_val & 0b01111111 | 0b10000000) : mantissa_val;
    int mantissa_shifted = mantissa_val >> (7 - mantissa_bits);
    int mantissa_round = mantissa_val >> (7 - mantissa_bits - 1);
    int round_bit = mantissa_round & 1;
    int round_condition = (round_bit == 1) && (mantissa_shifted != (1 << mantissa_bits) - 1);
    result.mantissa = round_condition ? mantissa_shifted + 1 : mantissa_shifted;

    return result;
}

__device__ float Converter_to_FP(int sign_c, int exponent_c, int mantissa_c, int exp_bits, int exp_offset) {
    const int bias = 127;

    if (exponent_c == 0xFF) {
        return sign_c ? -INFINITY : INFINITY;
    } else if (exponent_c == 0) {
        return sign_c ? -0.0f : 0.0f;
    }

    int shift_mul = __clz(mantissa_c) - 14;
    int mantissa_c_int = mantissa_c << (shift_mul + 1);
    int mantissa_mul_shift = (mantissa_c_int >> (18 - 7)) & 0x7F;

    if (mantissa_c_int & (1 << (18 - 8))) {
        mantissa_mul_shift++;
    }
    int exponent_mul_shift = 0;
    if (shift_mul!=18 && exponent_c !=0 && mantissa_c!=0){
        exponent_mul_shift = exponent_c - shift_mul ;
        if (exp_bits == 0){
           exponent_mul_shift += exp_offset + 3;
        }
        else {
            exponent_mul_shift += 3  + (128 - (1 << (exp_bits - 1))); 
        }
    }

    float mantissa = (1.0f + mantissa_mul_shift * powf(2.0f, -7));
    float result = (sign_c == 0 ? 1.0f : -1.0f) * ldexpf(mantissa, exponent_mul_shift - bias);

    return result;
}


__device__ void multiply(int index_a, int index_b, DecomposedFloat converted_a, DecomposedFloat converted_b, 
                         int exp_bits, int man_bits,
                         int& mantissa_mul, int& sign_mul, int& exp_mul, int exp_offset) {
    int mantissa_bits =  man_bits + 1 ;
    int mantissa_mask = (1 << mantissa_bits) - 1;  // Mask to extract mantissa_bits
    int implicit_bit = (1 << mantissa_bits);
    int mantissa_explicit_a = (converted_a.exponent != 0 && converted_a.mantissa != 0? implicit_bit : 0) | (converted_a.mantissa & mantissa_mask);
    int mantissa_explicit_b = (converted_b.exponent != 0 && converted_b.mantissa != 0? implicit_bit : 0) | (converted_b.mantissa & mantissa_mask);

    int exponent_offset = exp_bits == 0 ? 127 - exp_offset : (1 << (exp_bits - 1)) - 1;
    int min_exp = 0;
    int max_exp = exp_bits == 0 ? 31: (1 << exp_bits) - 1;
    // Handling NaN (Not a Number)
    bool is_nan_a = (converted_a.exponent == max_exp && converted_a.mantissa != 0);
    bool is_nan_b = (converted_b.exponent == max_exp && converted_b.mantissa != 0);
    if (is_nan_a || is_nan_b) {
        exp_mul = max_exp;
        mantissa_mul = 1;  // Non-zero mantissa for NaN
        sign_mul = 0;  // Sign bit is usually ignored for NaN
        return;
    }
    // Handling Infinity
    bool is_inf_a = (converted_a.exponent == max_exp && converted_a.mantissa == 0);
    bool is_inf_b = (converted_b.exponent == max_exp && converted_b.mantissa == 0);
    if (is_inf_a || is_inf_b) {
        exp_mul = max_exp;
        mantissa_mul = 0;
        sign_mul = converted_a.sign ^ converted_b.sign;
        return;
    }
    // Handling Zero
    bool is_zero_a = (converted_a.exponent == 0);
    bool is_zero_b = (converted_b.exponent == 0);
    if (is_zero_a || is_zero_b) {
        exp_mul = 0;
        mantissa_mul = 0;
        sign_mul = 0;
        return;
    }
    // Regular multiplication for non-special cases
    sign_mul = converted_a.sign ^ converted_b.sign;
    mantissa_mul = mantissa_explicit_a * mantissa_explicit_b;
    exp_mul = converted_a.exponent + converted_b.exponent - exponent_offset;

    // Check for overflow and underflow
    if (exp_mul > max_exp) {
        exp_mul = max_exp;  // Set to infinity
        mantissa_mul = 0;
    } else if (exp_mul < min_exp) {
        exp_mul = 0;  // Handle underflow
        mantissa_mul = 0;
    }
}


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
}
__device__ unsigned int concatenateTo10Bit(int sign, int exponent, int mantissa) {
    // Assuming sign is 1 bit, exponent is 6 bits, and mantissa is 3 bits
    // Shift and combine: [sign][exponent][mantissa]
    return (sign << 9) | (exponent << 3) | mantissa;
}


__global__ void convolutionKernel(int N, int C, int H, int W,
                                  int F, int HH, int WW,
                                  float* x, float* w,
                                //   __nv_bfloat16* out,
                                  float * out,
                                  int H_out, int W_out, 
                                  int pad, int stride, int exp_bits) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;


    int mantissa_bits = 6;

    if (w_out < W_out && h_out < H_out && f < F) {
        for (int n = 0; n < N; n++) {
            // Initialize accumulation variables for each output pixel
            int exp_sum = 0;
            int mantissa_sum = 0;
            int sign_sum = 0;
            int exp_offset = 100;
            for (int c = 0; c < C; c++) {
                for (int hh = 0; hh < HH; hh++) {
                    for (int ww = 0; ww < WW; ww++) {
                        int h_in = h_out * stride + hh - pad;
                        int w_in = w_out * stride + ww - pad;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            // Calculate index for input and filter
                            int index_a = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
                            int index_b = f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww;
                            DecomposedFloat FMAP = Floating2Binary_RFFP(&x[index_a], 8, 7);
                            DecomposedFloat WEIGHT = Floating2Binary_RFFP(&w[index_b], 8, 7);

                            DecomposedFloat converted_FMAP = RFFP_CONVERTER(&FMAP.sign, &FMAP.exponent, &FMAP.mantissa, exp_bits,  mantissa_bits, exp_offset);
                            DecomposedFloat converted_WEIGHT = RFFP_CONVERTER(&WEIGHT.sign, &WEIGHT.exponent, &WEIGHT.mantissa, exp_bits, mantissa_bits, exp_offset);

                            // Initialize multiplication and accumulation variables
                            int mantissa_mul, sign_mul, exp_mul;

                            // Call multiply function with converted values
                            multiply(index_a, index_b, converted_FMAP, converted_WEIGHT, exp_bits, mantissa_bits, mantissa_mul, sign_mul, exp_mul, exp_offset);

                            // Accumulate results
                            accumulate(exp_sum, mantissa_sum, sign_sum, exp_mul, mantissa_mul, sign_mul);
                        }
                    }
                }
            }

            // Write the final accumulated values to the output
            int index_out = n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out;
            // __nv_bfloat16 convertedValue;
            float convertedValue;
            convertedValue = Converter_to_FP(sign_sum, exp_sum, mantissa_sum, exp_bits, exp_offset);
            out[index_out]  += convertedValue;  
        }
    }
}
extern "C" {
    void conv2d(int N, int C, int H, int W,
        int F, int HH, int WW,
        float* x, float* w,
        // __nv_bfloat16* out,
        float * out,
        int pad, int stride, int exp_bits) {

        int H_out = 1 + (H + 2 * pad - HH) / stride;
        int W_out = 1 + (W + 2 * pad - WW) / stride;

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim(16, 16, 1);  // Adjust block dimensions as needed
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, (H_out + blockDim.y - 1) / blockDim.y, F);


        convolutionKernel<<<gridDim, blockDim>>>(N, C, H, W, F, HH, WW, x, w, out, H_out, W_out, pad, stride, exp_bits);
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }

        // Synchronize
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
            return;
        }
    }
}

__global__ void convolutionBackwardKernel_dw(int N_nb, int C_nb, int H_nb, int W_nb,
    float* x_nb, int F_nb, int HH_nb, int WW_nb, float* dout_nb, float* dw_nb, int H_dout_nb, int W_dout_nb, int stride_nb, int pad_nb, int exp_bits) {
    
    int f_nb = blockIdx.z; // filter index
    int c_nb = blockIdx.y; // channel index
    int hh_nb = blockIdx.x * blockDim.x + threadIdx.x; // height index
    int ww_nb = threadIdx.y; // width index
    


    if (f_nb < F_nb && c_nb < C_nb && hh_nb < HH_nb && ww_nb < WW_nb) {
        int exp_sum = 0;
        int mantissa_sum = 0;
        int sign_sum = 0;
        int mantissa_bits = 6;
        int exp_offset = 114;
        for (int n_nb = 0; n_nb < N_nb; n_nb++) {
            for (int h_nb = 0; h_nb < H_dout_nb; h_nb++) {
                for (int w_nb = 0; w_nb < W_dout_nb; w_nb++) {
                    int h_x_nb = h_nb * stride_nb + hh_nb - pad_nb;
                    int w_x_nb = w_nb * stride_nb + ww_nb - pad_nb;
                    if (h_x_nb >= 0 && h_x_nb < H_nb && w_x_nb >= 0 && w_x_nb < W_nb) {
                        int index_a = n_nb * (C_nb * H_nb * W_nb) + c_nb * (H_nb * W_nb) + h_x_nb * W_nb + w_x_nb;
                        int index_b = n_nb * (F_nb * H_dout_nb * W_dout_nb) + f_nb * (H_dout_nb * W_dout_nb) + h_nb * W_dout_nb + w_nb;
                        
                        DecomposedFloat FMAP = Floating2Binary_RFFP(&x_nb[index_a], 8, 7);
                        DecomposedFloat WEIGHT = Floating2Binary_RFFP(&dout_nb[index_b], 8, 7);

                        DecomposedFloat converted_FMAP = RFFP_CONVERTER(&FMAP.sign, &FMAP.exponent, &FMAP.mantissa, exp_bits, mantissa_bits, exp_offset);
                        DecomposedFloat converted_WEIGHT = RFFP_CONVERTER(&WEIGHT.sign, &WEIGHT.exponent, &WEIGHT.mantissa, exp_bits, mantissa_bits, exp_offset);
                        
                        int mantissa_mul, sign_mul, exp_mul;
                        // Call multiply function with converted values
                        multiply(index_a, index_b, converted_FMAP, converted_WEIGHT, exp_bits, mantissa_bits, mantissa_mul, sign_mul, exp_mul, exp_offset);

                        // Accumulate results
                        accumulate(exp_sum, mantissa_sum, sign_sum, exp_mul, mantissa_mul, sign_mul);

                    }
                }
            }
        }
        int index_out = f_nb * (C_nb * HH_nb * WW_nb) + c_nb * (HH_nb * WW_nb) + hh_nb * WW_nb + ww_nb;
        float convertedValue;
        convertedValue = Converter_to_FP(sign_sum, exp_sum, mantissa_sum, exp_bits, exp_offset);
        dw_nb[index_out]  += convertedValue; 
    }
}

extern "C" {
    void conv2d_backward_dw(int N_nb, int C_nb, int H_nb, int W_nb,
        float* x_nb, int F_nb, int HH_nb, int WW_nb, float* dout_nb, float* dw_nb, int H_dout_nb, int W_dout_nb, int stride_nb, int pad_nb, int exp_bits) {

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim_nb(16, 16, 1);  // Adjust block dimensions as needed
        dim3 gridDim_nb((HH_nb + blockDim_nb.x - 1) / blockDim_nb.x, C_nb, F_nb);

        // Launch CUDA kernel for backward convolution with respect to weights
        convolutionBackwardKernel_dw<<<gridDim_nb, blockDim_nb>>>(N_nb, C_nb, H_nb, W_nb, x_nb, F_nb, HH_nb, WW_nb, dout_nb, dw_nb, H_dout_nb, W_dout_nb, stride_nb, pad_nb, exp_bits);
    }
}

__global__ void convolutionBackwardKernel_db(int N, int F, int H_dout, int W_dout,
    float* dout, float* db, int exp_bits) {

    int f = blockIdx.x * blockDim.x + threadIdx.x;

    if (f < F) {
        float grad_db = 0.0;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H_dout; h++) {
                for (int w = 0; w < W_dout; w++) {
                    int index = n * (F * H_dout * W_dout) + f * (H_dout * W_dout) + h * W_dout + w;
                    grad_db += dout[index];
                }
            }
        }
        db[f] = grad_db;
    }
}

extern "C" {
    void conv2d_backward_db(int N, int F, int H_dout, int W_dout,
        float* dout, float* db, int exp_bits) {

        // Define grid and block dimensions for CUDA threads
        int blockSize = 256;  // Can be tuned based on GPU architecture
        int gridSize = (F + blockSize - 1) / blockSize;

        // Launch CUDA kernel for backward convolution with respect to bias
        convolutionBackwardKernel_db<<<gridSize, blockSize>>>(N, F, H_dout, W_dout, dout, db, exp_bits);
    }
}

__global__ void convolutionKernel_WB(int N, int C, int H, int W,
    float* input, int F, int HH, int WW,
    float* kernel, float* bias, float* output, int H_out, int W_out, int pad, int stride, int exp_bits) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (w_out < W_out && h_out < H_out && f < F) {
        for (int n = 0; n < N; n++) {
            int exp_sum = 0;
            int mantissa_sum = 0;
            int sign_sum = 0;
            int mantissa_bits = 6;
            int exp_offset = 114;

            for (int c = 0; c < C; c++) {
                for (int hh = 0; hh < HH; hh++) {
                    for (int ww = 0; ww < WW; ww++) {
                        int h_in = h_out * stride + hh - pad;
                        int w_in = w_out * stride + ww - pad;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            int index_a = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
                            int index_b = f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww;
                            DecomposedFloat FMAP = Floating2Binary_RFFP(&input[index_a], 8, 7);
                            DecomposedFloat WEIGHT = Floating2Binary_RFFP(&kernel[index_b], 8, 7);

                            DecomposedFloat converted_FMAP = RFFP_CONVERTER(&FMAP.sign, &FMAP.exponent, &FMAP.mantissa, exp_bits, mantissa_bits, exp_offset);
                            DecomposedFloat converted_WEIGHT = RFFP_CONVERTER(&WEIGHT.sign, &WEIGHT.exponent, &WEIGHT.mantissa, exp_bits, mantissa_bits, exp_offset);
                            // Initialize multiplication and accumulation variables
                            int mantissa_mul, sign_mul, exp_mul;

                            // Call multiply function with converted values
                            multiply(index_a, index_b, converted_FMAP, converted_WEIGHT, exp_bits, mantissa_bits, mantissa_mul, sign_mul, exp_mul, exp_offset);

                            // Accumulate results
                            accumulate(exp_sum, mantissa_sum, sign_sum, exp_mul, mantissa_mul, sign_mul);
                        }
                    }
                }
            }
            DecomposedFloat BIAS = Floating2Binary_RFFP(&bias[f], 8, 7);
            DecomposedFloat converted_BIAS = RFFP_CONVERTER(&BIAS.sign, &BIAS.exponent, &BIAS.mantissa, exp_bits, mantissa_bits, exp_offset);
            accumulate(converted_BIAS.sign, converted_BIAS.mantissa, converted_BIAS.exponent, exp_sum, mantissa_sum, sign_sum);
            int index_out = n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out;

            float convertedValue;
            convertedValue = Converter_to_FP(sign_sum, exp_sum, mantissa_sum, exp_bits, exp_offset);
            output[index_out]  += convertedValue;  
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
    }
}
