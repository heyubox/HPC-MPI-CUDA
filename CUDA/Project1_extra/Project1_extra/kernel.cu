#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
//CUDA RunTime API
#include <cuda_runtime.h>

//1M
#define DATA_SIZE 32767+1

#define THREAD_NUM 128
#define BLOCK_NUM 64

long int data[DATA_SIZE];


void GenerateNumbers(long int* number, long int size)
{
    for (long int i = 0; i < size; i++) {
        number[i] = i;
    }
}

long int fcpu(long int x)
{
    long int sum = 0;
    long int b = x, e = b + x;
    for (long int i = b; i <= e; ++i)
        sum +=int(sin(double(i)) * 2);
    return sum;
}

__device__ long int f(long int x)
{
    long int sum = 0;
    long int b = x, e = b + x;
    for (long int i = b; i <= e; ++i)
        sum += int(sin(double(i)) * 2);
    return sum;
}

__global__ static void sumOfF(long int* num, long int* result)
{

    //声明一块共享内存
    extern __shared__ long int shared[THREAD_NUM];

    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;

    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;

    shared[tid] = 0;

    int i;



    //thread需要同时通过tid和bid来确定
    for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {

        shared[tid] += f(num[i]);

    }

    __syncthreads();

    //树状加法
    int offset = 1, mask = 1;

    while (offset < THREAD_NUM)
    {
        if ((tid & mask) == 0)
        {
            shared[tid] += shared[tid + offset];
        }

        offset += offset;
        mask = offset + mask;
        __syncthreads();

    }
    if (tid == 0)
    {
        result[bid] = shared[0];
    }

}



int main()
{



    //生成随机数
    GenerateNumbers(data, DATA_SIZE);

    /*把数据复制到显卡内存中*/
    long int* gpudata, * result;


    //cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果，time用来存储运行时间 )
    cudaMalloc((void**)&gpudata, sizeof(long int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(long int) * BLOCK_NUM);


    //cudaMemcpy 将产生的随机数复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(gpudata, data, sizeof(long int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    sumOfF << < BLOCK_NUM, THREAD_NUM, 0 >> > (gpudata, result);


    /*把结果从显示芯片复制回主内存*/

    long int sum[THREAD_NUM * BLOCK_NUM];

    clock_t time_use[BLOCK_NUM * 2];

    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(&sum, result, sizeof(long int) * THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
    

    //Free
    cudaFree(gpudata);
    cudaFree(result);


    long int final_sum = 0;

    for (int i = 0; i < BLOCK_NUM; i++) {

        final_sum += sum[i];

    }


    printf("GPUsum: %d \n", final_sum);





    final_sum = 0;

    for (int i = 0; i < DATA_SIZE; i++) {

        final_sum += fcpu(data[i]);

    }

    printf("CPUsum: %d \n", final_sum-1);

    return 0;
}