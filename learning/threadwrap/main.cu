#include <cstdio>
#include <iostream>
#include <chrono>

#include "cuda_runtime_api.h"

#define NUM 1000

#define GPU_CHECK(ans)                                                         \
    { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
};

__global__ void tst01() {
    int threadIdX = threadIdx.x;  // 当前线程在 x 维度上的索引
    int counter = 0;
    // printf("%d",threadIdX);
    for(int i = 0; i < NUM; i++){
        if(threadIdX%2){
            counter = i;
        }
    }
    printf("%d\n",counter);
}

__global__ void tst02() {
    int threadIdX = threadIdx.x;  // 当前线程在 x 维度上的索引
    int counter = 0;
    // printf("%d",threadIdX);
    for(int i = 0; i < NUM; i++){
        if(threadIdX<100){
            counter = i;
        }
    }
    printf("%d\n",counter);
}

int main(int argc, char **argv) {

    tst01<<<1, 32>>>();
    // Find/set device and get device properties
    int device = 0;
    cudaDeviceProp deviceProp;
    GPU_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    if (!(deviceProp.major > 3 ||
          (deviceProp.major == 3 && deviceProp.minor >= 5))) {
        printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting.",
            device, deviceProp.name);
        return 0;
    }

    // 获取当前时间点
    auto start = std::chrono::steady_clock::now();

    // Execute
    std::cout << "Running tst01: " << std::endl;
    tst01<<<1, 32>>>();
    cudaDeviceSynchronize();

    // 获取结束时间点
    auto end_0 = std::chrono::steady_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_0 - start);

    // 输出程序执行时间
    std::cout << "\nexe time: " << duration.count() << " ms" << std::endl;

    // Execute
    std::cout << "Running tst02: " << std::endl;
    tst02<<<1, 32>>>();
    cudaDeviceSynchronize();

    // 获取结束时间点
    auto end_1 = std::chrono::steady_clock::now();
    // 计算时间差
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - end_0);
    // 输出程序执行时间
    std::cout << "\nexe time: " << duration.count() << " ms" << std::endl;

    return 0;
}