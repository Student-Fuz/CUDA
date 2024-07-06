#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void kernel(int* data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 写入共享内存
    __shared__ int flag;
	
    flag = 0;
	
	if(tid == 0){
		for(int i = 0; i < 10000; i++);
		flag = 1;
		//__threadfence_block();
	}
	
    __threadfence_block();
	
	while(!flag){
	}
    
    // 打印结果
    printf("Thread %d\n", tid);
}

int main()
{
    int numThreads = 5;
    int numBlocks = 1;
    
    // 启动内核
    kernel<<<numBlocks, numThreads>>>(nullptr);
    
    // 同步设备
    cudaDeviceSynchronize();
	getchar();
    
    return 0;
}