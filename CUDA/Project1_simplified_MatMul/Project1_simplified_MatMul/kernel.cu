#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
using namespace std;
//CUDA RunTime API
#include <cuda_runtime.h>

#define TILE_WIDTH 16




//�����������
void matgen(int* a, int n)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {

            a[i * n + j] = (int)rand() / RAND_MAX + (int)rand() / (RAND_MAX * RAND_MAX);

        }
    }
}

void matgen(int* a, int c, int r)
{
    int i, j;

    for (i = 0; i < c; i++)
    {
        for (j = 0; j < r; j++)
        {

            a[i * c + j] = (int)rand() % 10;

        }
    }
}

template <size_t BLOCK_SIZE>
void __global__ MatMatMul(
    const int* A,
    const int* B,
    int* C,
    const size_t m,//a_r
    const size_t n,//b_r a_c
    const size_t k)//b_c
{
    int bx = blockIdx.x;		int by = blockIdx.y;
    int tx = threadIdx.x;		int ty = threadIdx.y;

    //ȷ����������е��к���
    int row = by * BLOCK_SIZE + ty;
    int column = bx * BLOCK_SIZE + tx;


    if (row < m && column < k)
    {
        int t = 0;

        for (int i = 0; i < n; i++)
        {
            t += A[row * n + i] * B[i * n + column];
        }
        C[row * n + column] = t;
    }

}







int main()
{



    //�������
    int* a, * b, * c, * d;

    const size_t n = 1 << 12;


    const size_t a_r = n, a_c = n, b_r = n, b_c = n;

    //�����ڴ�
    a = (int*)malloc(sizeof(int) * n * n);
    b = (int*)malloc(sizeof(int) * n * n);
    c = (int*)malloc(sizeof(int) * n * n);
    d = (int*)malloc(sizeof(int) * n * n);

    //�������������
    srand(0);

    //������ɾ���
    matgen(a, a_r, a_c);
    matgen(b, b_r, b_c);

    /*�����ݸ��Ƶ��Կ��ڴ���*/
    int* cuda_a, * cuda_b, * cuda_c;

    

    //cudaMalloc ȡ��һ���Կ��ڴ� 
    cudaMalloc((void**)&cuda_a, sizeof(int) * a_r * a_c);
    cudaMalloc((void**)&cuda_b, sizeof(int) * b_r * b_c);
    cudaMalloc((void**)&cuda_c, sizeof(int) * a_r * b_c);


    //cudaMemcpy �������ľ����Ƶ��Կ��ڴ���
    //cudaMemcpyHostToDevice - ���ڴ渴�Ƶ��Կ��ڴ�
    //cudaMemcpyDeviceToHost - ���Կ��ڴ渴�Ƶ��ڴ�
    cudaMemcpy(cuda_a, a, sizeof(int) * a_r * a_c, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(int) * b_r * b_c, cudaMemcpyHostToDevice);



    //��������ṹ
    cout << "n1*n2:" << n << "*" << n << endl;
    cout << "TILE_WIDTH:" << TILE_WIDTH << endl;
    cout << "dimGrid:" << (b_c - 1) / TILE_WIDTH + 1 << ',' << (a_r - 1) / TILE_WIDTH + 1 << ',' << 1 << endl;
    cout << "dimBlock:" << TILE_WIDTH << ',' << TILE_WIDTH << ',' << 1 << endl;
    dim3 dimGrid((b_c - 1) / TILE_WIDTH + 1, (a_r - 1) / TILE_WIDTH + 1, 1);	//����ȡ��
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    clock_t sp = clock();
    cout << "start of CUDA:" << sp << endl;

    // ��CUDA ��ִ�к��� �﷨����������<<<block ��Ŀ, thread ��Ŀ, shared memory ��С>>>(����...);
    MatMatMul<TILE_WIDTH> << <dimGrid, dimBlock >> > (cuda_a, cuda_b, cuda_c, a_r, a_c/*b_r*/, b_c);


    //cudaMemcpy ��������Դ��и��ƻ��ڴ�
    cudaMemcpy(c, cuda_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
    //cudaMemcpy(&time_use, time, sizeof(clock_t) * blocks_num * 2, cudaMemcpyDeviceToHost);


    clock_t ep = clock();
    cout << "end of CUDA and start of CPU:" << ep << endl;
    cout << "cost:" << (double)(ep - sp) / CLOCKS_PER_SEC;
    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    cudaFree(time);


//��֤��ȷ���뾫ȷ��
    /*
    //CPU����˷����������d
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double t = 0;

            for (int k = 0; k < n; k++)
            {

                t += a[i * n + k] * b[k * n + j];

            }

            d[i * n + j] = t;

        }
    }
*/
    





    return 0;

}

