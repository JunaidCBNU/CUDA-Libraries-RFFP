#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>  // Include the CUDA runtime header

__global__ void convolutionKernel(int N, int C, int H, int W,
    float* input, int F, int HH, int WW,
    float* kernel, float* output, int H_out, int W_out, int pad, int stride) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (w_out < W_out && h_out < H_out && f < F) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0;

            for (int c = 0; c < C; c++) {
                for (int hh = 0; hh < HH; hh++) {
                    for (int ww = 0; ww < WW; ww++) {
                        int h_in = h_out * stride + hh - pad;
                        int w_in = w_out * stride + ww - pad;

                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            sum += input[n * (C * H * W) + c * (H * W) + h_in * W + w_in] *
                                   kernel[f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww];
                        }
                    }
                }
            }

            output[n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out] += sum;
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
            float sum = 0.0;

            for (int c = 0; c < C; c++) {
                for (int hh = 0; hh < HH; hh++) {
                    for (int ww = 0; ww < WW; ww++) {
                        int h_in = h_out * stride + hh - pad;
                        int w_in = w_out * stride + ww - pad;

                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            sum += input[n * (C * H * W) + c * (H * W) + h_in * W + w_in] *
                                   kernel[f * (C * HH * WW) + c * (HH * WW) + hh * WW + ww];
                        }
                    }
                }
            }

            // Add bias term for the current filter
            sum += bias[f];

            output[n * (F * H_out * W_out) + f * (H_out * W_out) + h_out * W_out + w_out] += sum;
        }
    }
}
extern "C" {
    void conv2d_WB(int N, int C, int H, int W,
        float* input, int F, int HH, int WW,
        float* kernel, float* bias, float* output, int pad, int stride) {

        int H_out = 1 + (H + 2 * pad - HH) / stride;
        int W_out = 1 + (W + 2 * pad - WW) / stride;

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim(16, 16, 1);
        dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x, (H_out + blockDim.y - 1) / blockDim.y, F);

        // Launch CUDA kernel for convolution
        convolutionKernel_WB<<<gridDim, blockDim>>>(N, C, H, W, input, F, HH, WW, kernel, bias, output, H_out, W_out, pad, stride);
    }
}


// Back Propagation

__global__ void convolutionBackwardKernel_dw(int N_nb, int C_nb, int H_nb, int W_nb,
    float* x_nb, int F_nb, int HH_nb, int WW_nb, float* dout_nb, float* dw_nb, int H_dout_nb, int W_dout_nb, int stride_nb, int pad_nb) {
    
    int f_nb = blockIdx.z; // filter index
    int c_nb = blockIdx.y; // channel index
    int hh_nb = blockIdx.x * blockDim.x + threadIdx.x; // height index
    int ww_nb = threadIdx.y; // width index

    if (f_nb < F_nb && c_nb < C_nb && hh_nb < HH_nb && ww_nb < WW_nb) {
        float grad_dw_nb = 0.0;
        for (int n_nb = 0; n_nb < N_nb; n_nb++) {
            for (int h_nb = 0; h_nb < H_dout_nb; h_nb++) {
                for (int w_nb = 0; w_nb < W_dout_nb; w_nb++) {
                    int h_x_nb = h_nb * stride_nb + hh_nb - pad_nb;
                    int w_x_nb = w_nb * stride_nb + ww_nb - pad_nb;
                    if (h_x_nb >= 0 && h_x_nb < H_nb && w_x_nb >= 0 && w_x_nb < W_nb) {
                        grad_dw_nb += x_nb[n_nb * (C_nb * H_nb * W_nb) + c_nb * (H_nb * W_nb) + h_x_nb * W_nb + w_x_nb] *
                                   dout_nb[n_nb * (F_nb * H_dout_nb * W_dout_nb) + f_nb * (H_dout_nb * W_dout_nb) + h_nb * W_dout_nb + w_nb];
                    }
                }
            }
        }
        dw_nb[f_nb * (C_nb * HH_nb * WW_nb) + c_nb * (HH_nb * WW_nb) + hh_nb * WW_nb + ww_nb] = grad_dw_nb;
    }
}

extern "C" {
    void conv2d_backward_dw(int N_nb, int C_nb, int H_nb, int W_nb,
        float* x_nb, int F_nb, int HH_nb, int WW_nb, float* dout_nb, float* dw_nb, int H_dout_nb, int W_dout_nb, int stride_nb, int pad_nb) {

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim_nb(16, 16, 1);  // Adjust block dimensions as needed
        dim3 gridDim_nb((HH_nb + blockDim_nb.x - 1) / blockDim_nb.x, C_nb, F_nb);

        // Launch CUDA kernel for backward convolution with respect to weights
        convolutionBackwardKernel_dw<<<gridDim_nb, blockDim_nb>>>(N_nb, C_nb, H_nb, W_nb, x_nb, F_nb, HH_nb, WW_nb, dout_nb, dw_nb, H_dout_nb, W_dout_nb, stride_nb, pad_nb);
    }
}

#include <cuda_runtime.h>

__global__ void convolutionBackwardKernel_db(int N, int F, int H_dout, int W_dout,
    float* dout, float* db) {
    
    int f = blockIdx.x * blockDim.x + threadIdx.x;

    if (f < F) {
        float grad_db = 0.0;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H_dout; h++) {
                for (int w = 0; w < W_dout; w++) {
                    grad_db += dout[n * (F * H_dout * W_dout) + f * (H_dout * W_dout) + h * W_dout + w];
                }
            }
        }
        db[f] = grad_db;
    }
}

extern "C" {
    void conv2d_backward_db(int N, int F, int H_dout, int W_dout,
        float* dout, float* db) {

        // Define grid and block dimensions for CUDA threads
        int blockSize = 256;  // Can be tuned based on GPU architecture
        int gridSize = (F + blockSize - 1) / blockSize;

        // Launch CUDA kernel for backward convolution with respect to bias
        convolutionBackwardKernel_db<<<gridSize, blockSize>>>(N, F, H_dout, W_dout, dout, db);
    }
}


__global__ void convolutionBackwardKernel_dx(int N_wb, int C_wb, int H_wb, int W_wb,
    float* dout_wb, int F_wb, int HH_wb, int WW_wb, float* w_wb, float* dx_wb, int H_dout_wb, int W_dout_wb, int stride_wb, int pad_wb) 
    {
    // if (threadIdx.x < 5) {
    //     if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //         printf("dout_wb[%d]: %.10e\n", threadIdx.x, dout_wb[threadIdx.x]);
    //         printf("w_wb[%d]: %.10e\n", threadIdx.x, w_wb[threadIdx.x]);
    //     }
    // }
    int n = blockIdx.z; // batch index
    int c = blockIdx.y; // channel index
    int h = blockIdx.x * blockDim.x + threadIdx.x; // height index
    int w_idxx = threadIdx.y; // width index (renamed to avoid conflict)

    if (n < N_wb && c < C_wb && h < H_wb && w_idxx < W_wb) {
        float grad_dx_wb = 0.0;
        for (int f = 0; f < F_wb; f++) {
            for (int hh = 0; hh < HH_wb; hh++) {
                for (int ww = 0; ww < WW_wb; ww++) {
                    int h_dout = (h + pad_wb - hh) / stride_wb;
                    int w_dout = (w_idxx + pad_wb - ww) / stride_wb;
                    if (h_dout >= 0 && h_dout < H_dout_wb && w_dout >= 0 && w_dout < W_dout_wb) {
                        grad_dx_wb += w_wb[f * (C_wb * HH_wb * WW_wb) + c * (HH_wb * WW_wb) + hh * WW_wb + ww] * 
                                dout_wb[n * (F_wb * H_dout_wb * W_dout_wb) + f * (H_dout_wb * W_dout_wb) + h_dout * W_dout_wb + w_dout];
                    }
                }
            }
        }
        dx_wb[n * (C_wb * H_wb * W_wb) + c * (H_wb * W_wb) + h * W_wb + w_idxx] = grad_dx_wb;
        // printf(" %.10e\n",grad_dx_wb);
        // if (threadIdx.x < 25 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("dx_nb[%d, %d, %d, %d]: %.10e\n", n, c, h, w_idxx, grad_dx_wb);
        // }

    }
}
extern "C" {
    void conv2d_backward_dx(int N_wb, int C_wb, int H_wb, int W_wb,
        float* dout_wb, int F_wb, int HH_wb, int WW_wb, float* w_wb, float* dx_wb, int H_dout_wb, int W_dout_wb, int stride_wb, int pad_wb) {

        // Define grid and block dimensions for CUDA threads
        dim3 blockDim(16, 16, 1);  // Adjust block dimensions as needed
        dim3 gridDim((H_wb + blockDim.x - 1) / blockDim.x, C_wb, N_wb);

        // Launch CUDA kernel for backward convolution with respect to input
        convolutionBackwardKernel_dx<<<gridDim, blockDim>>>(N_wb, C_wb, H_wb, W_wb, dout_wb, F_wb, HH_wb, WW_wb, w_wb, dx_wb, H_dout_wb, W_dout_wb, stride_wb, pad_wb);
    }
}

#include <cfloat>

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

