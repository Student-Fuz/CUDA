#include <cstdio>
#include <iostream>
#include <chrono>
#include <cudnn.h>

#include "cuda_runtime_api.h"

// 多核加法
__global__
void add(int n, float *x, float *y)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

// 单核加法
void CPU_add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<28; // 1M elements

    // Allocate host memory
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));

    //*********************************************************************
    // CPU RUN
    //*********************************************************************
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // 获取当前时间点
    auto timePoint_0 = std::chrono::steady_clock::now();

    CPU_add(N, h_x, h_y);

    // 获取当前时间点
    auto timePoint_1 = std::chrono::steady_clock::now();

    // 计算时间差
    auto duration_0 = std::chrono::duration_cast<std::chrono::milliseconds>(timePoint_1 - timePoint_0);
    // 输出程序执行时间
    std::cout << "CPU exe time: " << duration_0.count() << " ms" << std::endl;

    //*********************************************************************
    // GPU END
    //*********************************************************************

    //*********************************************************************
    // GPU RUN
    //*********************************************************************
    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // 获取当前时间点
    auto timePoint_2 = std::chrono::steady_clock::now();

    // Run kernel on 1M elements on the GPU
    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize; // 向上取整
    //或写作
    // int numBlocks = (N - 1) / blockSize + 1; // 向上取整
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // 获取当前时间点
    auto timePoint_3 = std::chrono::steady_clock::now();

    // 计算时间差
    auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(timePoint_3 - timePoint_2);
    // 输出程序执行时间
    std::cout << "GPU exe time: " << duration_1.count() << " ms" << std::endl;

    // Copy data back to host
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    //*********************************************************************
    // GPU END
    //*********************************************************************

    // Free memory
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
