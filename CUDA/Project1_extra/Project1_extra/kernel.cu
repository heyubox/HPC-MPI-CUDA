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

    //����һ�鹲���ڴ�
    extern __shared__ long int shared[THREAD_NUM];

    //��ʾĿǰ�� thread �ǵڼ��� thread���� 0 ��ʼ���㣩
    const int tid = threadIdx.x;

    //��ʾĿǰ�� thread ���ڵڼ��� block���� 0 ��ʼ���㣩
    const int bid = blockIdx.x;

    shared[tid] = 0;

    int i;



    //thread��Ҫͬʱͨ��tid��bid��ȷ��
    for (i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {

        shared[tid] += f(num[i]);

    }

    __syncthreads();

    //��״�ӷ�
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



    //���������
    GenerateNumbers(data, DATA_SIZE);

    /*�����ݸ��Ƶ��Կ��ڴ���*/
    long int* gpudata, * result;


    //cudaMalloc ȡ��һ���Կ��ڴ� ( ����result�����洢��������time�����洢����ʱ�� )
    cudaMalloc((void**)&gpudata, sizeof(long int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(long int) * BLOCK_NUM);


    //cudaMemcpy ����������������Ƶ��Կ��ڴ���
    //cudaMemcpyHostToDevice - ���ڴ渴�Ƶ��Կ��ڴ�
    //cudaMemcpyDeviceToHost - ���Կ��ڴ渴�Ƶ��ڴ�
    cudaMemcpy(gpudata, data, sizeof(long int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // ��CUDA ��ִ�к��� �﷨����������<<<block ��Ŀ, thread ��Ŀ, shared memory ��С>>>(����...);
    sumOfF << < BLOCK_NUM, THREAD_NUM, 0 >> > (gpudata, result);


    /*�ѽ������ʾоƬ���ƻ����ڴ�*/

    long int sum[THREAD_NUM * BLOCK_NUM];

    clock_t time_use[BLOCK_NUM * 2];

    //cudaMemcpy ��������Դ��и��ƻ��ڴ�
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