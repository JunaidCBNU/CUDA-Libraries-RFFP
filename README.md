Lets Practice Playing with Mantissa and Exponent Width along with CUSTOM EXPONENT BIAS.

Please download CUDA_PLAY on your PC

Compiler it with below command

nvcc -arch=sm_80  -G -g -o conv_play CUDA_PLAY.cu
./conv_play

In this CUDA_PLAY, i have added functionalities such as:
1. Convolution with Bias
2. Point Wise Convolution
3. Depth Wise Convolution

You can take bfloat reference pooling code from conv16.cu file and add in CUDA_PLAY.cu and perform pooling in custom format.

In similar fashion you can convert any module.
