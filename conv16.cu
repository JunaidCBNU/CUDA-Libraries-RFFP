#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>  // Include the CUDA runtime header
#include <cuda_bf16.h>
#include <cfloat>

__global__ void convolutionKernel(int N, int C, int H, int W,
    float* input, int F, int HH, int WW,
    float* kernel, float* output, int H_out, int W_out, int pad, int stride) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (w_out < W_out && h_out < H_out && f < F) {
        for (int n = 0; n < N; n++) {
            __nv_bfloat16 sum = __float2bfloat16(0.0);

            for (int c = 0; c < C; c++) {
                for (int hh = 0; hh < HH; hh++) {
                    for (int ww = 0; ww < WW; ww++) {
                        int h_in = h_out * stride + hh - pad;
                        int w_in = w_out * stride + ww - pad;

                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            __nv_bfloat16 input_bf16 = __float2bfloat16(input[n * (C * H * W) + c * (H * W) + h_in * W + w_in]);
                            __nv_bfloat16 kernel_bf16 = __float2bfloat16(kernel[f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww]);
                            sum = __hadd(sum, __hmul(input_bf16, kernel_bf16));
                        }
                    }
                }
            }

            output[n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out] =
                __bfloat162float(__hadd(__float2bfloat16(output[n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out]), sum));
        }
    }
}

extern "C" {
    void conv2d(int N, int C, int H, int W,
        float* input, int F, int HH, int WW,
        float* kernel, float* output, int pad, int stride) {

        int H_out = 1 + (H + 2 * pad - HH) / stride;
        int W_out = 1 + (W + 2 * pad - WW) / stride;

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim(16, 16, 1);  // Adjust block dimensions as needed
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, (H_out + blockDim.y - 1) / blockDim.y, F);

        // Launch CUDA kernel for convolution
        convolutionKernel<<<gridDim, blockDim>>>(N, C, H, W, input, F, HH, WW, kernel, output, H_out, W_out, pad, stride);
    }
}


// Convolution with Bias
__global__ void convolutionKernel_WB(int N, int C, int H, int W,
    float* input, int F, int HH, int WW,
    float* kernel, float* bias, float* output, int H_out, int W_out, int pad, int stride) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (w_out < W_out && h_out < H_out && f < F) {
        for (int n = 0; n < N; n++) {
            __nv_bfloat16 sum = __float2bfloat16(0.0);

            for (int c = 0; c < C; c++) {
                for (int hh = 0; hh < HH; hh++) {
                    for (int ww = 0; ww < WW; ww++) {
                        int h_in = h_out * stride + hh - pad;
                        int w_in = w_out * stride + ww - pad;

                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            __nv_bfloat16 input_val = __float2bfloat16(input[n * (C * H * W) + c * (H * W) + h_in * W + w_in]);
                            __nv_bfloat16 kernel_val = __float2bfloat16(kernel[f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww]);
                            sum = __hadd(sum, __hmul(input_val, kernel_val));
                        }
                    }
                }
            }

            __nv_bfloat16 bias_val = __float2bfloat16(bias[f]);
            sum = __hadd(sum, bias_val);
            output[n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out] += __bfloat162float(sum);
        }
    }
}

extern "C" {
    void conv2d_WB(int N, int C, int H, int W,
        float* input, int F, int HH, int WW,
        float* kernel, float* bias, float* output, int pad, int stride) {

        int H_out = 1 + (H + 2 * pad - HH) / stride;
        int W_out = 1 + (W + 2 * pad - WW) / stride;

        dim3 blockDim(16, 16, 1);
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, (H_out + blockDim.y - 1) / blockDim.y, F);

        convolutionKernel_WB<<<gridDim, blockDim>>>(N, C, H, W, input, F, HH, WW, kernel, bias, output, H_out, W_out, pad, stride);
    }
}



__global__ void convolutionBackwardKernel_dw(int N_nb, int C_nb, int H_nb, int W_nb,
    float* x_nb, int F_nb, int HH_nb, int WW_nb, float* dout_nb, float* dw_nb, int H_dout_nb, int W_dout_nb, int stride_nb, int pad_nb) {
    
    int f_nb = blockIdx.z;
    int c_nb = blockIdx.y;
    int hh_nb = blockIdx.x * blockDim.x + threadIdx.x;
    int ww_nb = threadIdx.y;

    if (f_nb < F_nb && c_nb < C_nb && hh_nb < HH_nb && ww_nb < WW_nb) {
        __nv_bfloat16 grad_dw_nb = __float2bfloat16(0.0);
        for (int n_nb = 0; n_nb < N_nb; n_nb++) {
            for (int h_nb = 0; h_nb < H_dout_nb; h_nb++) {
                for (int w_nb = 0; w_nb < W_dout_nb; w_nb++) {
                    int h_x_nb = h_nb * stride_nb + hh_nb - pad_nb;
                    int w_x_nb = w_nb * stride_nb + ww_nb - pad_nb;
                    if (h_x_nb >= 0 && h_x_nb < H_nb && w_x_nb >= 0 && w_x_nb < W_nb) {
                        __nv_bfloat16 x_val = __float2bfloat16(x_nb[n_nb * (C_nb * H_nb * W_nb) + c_nb * (H_nb * W_nb) + h_x_nb * W_nb + w_x_nb]);
                        __nv_bfloat16 dout_val = __float2bfloat16(dout_nb[n_nb * (F_nb * H_dout_nb * W_dout_nb) + f_nb * (H_dout_nb * W_dout_nb) + h_nb * W_dout_nb + w_nb]);
                        grad_dw_nb = __hadd(grad_dw_nb, __hmul(x_val, dout_val));
                    }
                }
            }
        }
        dw_nb[f_nb * (C_nb * HH_nb * WW_nb) + c_nb * (HH_nb * WW_nb) + hh_nb * WW_nb + ww_nb] = __bfloat162float(grad_dw_nb);
    }
}

extern "C" {
    void conv2d_backward_dw(int N_nb, int C_nb, int H_nb, int W_nb,
        float* x_nb, int F_nb, int HH_nb, int WW_nb, float* dout_nb, float* dw_nb, int H_dout_nb, int W_dout_nb, int stride_nb, int pad_nb) {

        dim3 blockDim_nb(16, 16, 1);
        dim3 gridDim_nb((HH_nb + blockDim_nb.x - 1) / blockDim_nb.x, C_nb, F_nb);

        convolutionBackwardKernel_dw<<<gridDim_nb, blockDim_nb>>>(N_nb, C_nb, H_nb, W_nb, x_nb, F_nb, HH_nb, WW_nb, dout_nb, dw_nb, H_dout_nb, W_dout_nb, stride_nb, pad_nb);
    }
}

__global__ void convolutionBackwardKernel_db(int N, int F, int H_dout, int W_dout,
    float* dout, float* db) {
    
    int f = blockIdx.x * blockDim.x + threadIdx.x;

    if (f < F) {
        __nv_bfloat16 grad_db = __float2bfloat16(0.0);
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H_dout; h++) {
                for (int w = 0; w < W_dout; w++) {
                    __nv_bfloat16 dout_val = __float2bfloat16(dout[n * (F * H_dout * W_dout) + f * (H_dout * W_dout) + h * W_dout + w]);
                    grad_db = __hadd(grad_db, dout_val);
                }
            }
        }
        db[f] = __bfloat162float(grad_db);
    }
}

extern "C" {
    void conv2d_backward_db(int N, int F, int H_dout, int W_dout,
        float* dout, float* db) {

        int blockSize = 256;
        int gridSize = (F + blockSize - 1) / blockSize;

        convolutionBackwardKernel_db<<<gridSize, blockSize>>>(N, F, H_dout, W_dout, dout, db);
    }
}


///\\\ ******* For MAX Pooling ********** \\\///
__global__ void max_pooling_forward_kernel(float *x, float *out, int *positions,
                                           int N, int C, int H, int W,
                                           int H_out, int W_out,
                                           int pool_height, int pool_width, int stride) {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index
    int totalOutputElements = N * C * H_out * W_out; // Total elements in output

    if (globalIndex < totalOutputElements) {
        // Calculate n, c, h_out, and w_out based on globalIndex
        int n = globalIndex / (C * H_out * W_out);
        int c = (globalIndex / (H_out * W_out)) % C;
        int h_out = (globalIndex / W_out) % H_out;
        int w_out = globalIndex % W_out;

        float max_val = -FLT_MAX;
        int max_index = -1;

        // Perform pooling operation
        for (int h = h_out * stride; h < min(h_out * stride + pool_height, H); h++) {
            for (int w = w_out * stride; w < min(w_out * stride + pool_width, W); w++) {
                int index = n * (C * H * W) + c * (H * W) + h * W + w;
                if (x[index] > max_val) {
                    max_val = x[index];
                    // Calculate relative position within 2x2 window
                    max_index = (h - h_out * stride) * pool_width + (w - w_out * stride);
                }
            }
        }

        // Write results to output tensors
        int out_index = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
        out[out_index] = max_val;
        positions[out_index] = max_index;
    }
}

extern "C" {
    void max_pooling_forward(int N, int C, int H, int W, 
                            float* x_ptr, float* out_ptr, int* pos_ptr, 
                            int pool_height, int pool_width, int stride) {
        int H_out = (H - pool_height) / stride + 1;
        int W_out = (W - pool_width) / stride + 1;
        int totalOutputElements = N * C * H_out * W_out;

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim(256);  // Adjust block dimensions as needed
        dim3 gridDim((totalOutputElements + blockDim.x - 1) / blockDim.x);

        // Launch CUDA kernel for max pooling forward
        max_pooling_forward_kernel<<<gridDim, blockDim>>>(x_ptr, out_ptr, pos_ptr, N, C, H, W, H_out, W_out, pool_height, pool_width, stride);
    }
}

__global__ void max_pooling_backward_kernel(const float *dout, float *dx, const int *positions,
                                            int N, int C, int H, int W,
                                            int H_out, int W_out,
                                            int pool_height, int pool_width, int stride) {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputElements = N * C * H_out * W_out;

    if (globalIndex < totalOutputElements) {
        int n = globalIndex / (C * H_out * W_out);
        int c = (globalIndex / (H_out * W_out)) % C;
        int h_out = (globalIndex / W_out) % H_out;
        int w_out = globalIndex % W_out;

        int pos_index = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
        int max_index = positions[pos_index];
        int h_max = h_out * stride + max_index / pool_width;
        int w_max = w_out * stride + max_index % pool_width;
        int index = n * (C * H * W) + c * (H * W) + h_max * W + w_max;

        dx[index] += dout[pos_index];
    }
}


extern "C" {
    void max_pooling_backward(int N, int C, int H, int W, 
                              const float* dout_ptr, float* dx_ptr, const int* pos_ptr, 
                              int pool_height, int pool_width, int stride) {
        int H_out = (H - pool_height) / stride + 1;
        int W_out = (W - pool_width) / stride + 1;
        int totalOutputElements = N * C * H_out * W_out;

        dim3 blockDim(256); // Adjust block dimensions as needed
        dim3 gridDim((totalOutputElements + blockDim.x - 1) / blockDim.x);

        max_pooling_backward_kernel<<<gridDim, blockDim>>>(dout_ptr, dx_ptr, pos_ptr, 
                                                           N, C, H, W, H_out, W_out, 
                                                           pool_height, pool_width, stride);
    }
}

