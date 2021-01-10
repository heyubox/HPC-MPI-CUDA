#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include<iostream>


#include <fstream>
using namespace std;
#define TILE_WIDTH 64
//�����������

void matgen(int* a, int c,int r)
{
	int i, j;

	for (i = 0; i < c; i++)
	{
		for (j = 0; j < r; j++)
		{

			a[i * c + j] = (int)rand() % 10 ;

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
	//���빲���ڴ棬������ÿ��block��
	__shared__ int ds_A[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int ds_B[BLOCK_SIZE][BLOCK_SIZE];

	
	int bx = blockIdx.x;		int by = blockIdx.y;
	int tx = threadIdx.x;		int ty = threadIdx.y;

	//ȷ����������е��к���
	int Row = by * BLOCK_SIZE + ty;
	int Col = bx * BLOCK_SIZE + tx;

	//��ʱ����
	int Cvalue = 0;

	//ѭ������A,B��Ƭ�����������󣬷ֽ׶ν��м���
	for (int t = 0; t < (n - 1) / BLOCK_SIZE + 1; ++t)//n = a_c,b_r
	{
		//��A,B������Ƭ���Ľ������shared memory�У�ÿ���̼߳�����Ӧ��CԪ�ص�A/B����Ԫ��
		//��Ƭ����һ�����ڣ���Ҫ�����A��B�����ٻ���
		if (Row < m && t * BLOCK_SIZE + tx < n)		//Խ�紦�����������С�ľ������
			//ds_A[tx][ty] = A[t*TILE_WIDTH + tx][Row];
			ds_A[tx][ty] = A[Row * n + t * BLOCK_SIZE + tx];//�Ժϲ��ķ�ʽ������Ƭ
		else
			ds_A[tx][ty] = 0.0;

		if (t * BLOCK_SIZE + ty < n && Col < k)
			//ds_B[tx][ty] = B[Col][t*TILE_WIDTH + ty];
			ds_B[tx][ty] = B[(t * BLOCK_SIZE + ty) * k + Col];
		else
			ds_B[tx][ty] = 0.0;

		//��֤tile�����е�Ԫ�ر�����
		__syncthreads();

		for (int i = 0; i < BLOCK_SIZE; ++i)
			Cvalue += ds_A[i][ty] * ds_B[tx][i];//��shared memory��ȡֵ

		//ȷ�������߳���ɼ���󣬽�����һ���׶εļ���
		__syncthreads();

		if (Row < m && Col < k)
			C[Row * k + Col] = Cvalue;
	}

}


int main()
{
	const size_t n =1<<11;
	

	const size_t a_r = n, a_c= n, b_r= n, b_c = n;
	
	int* a, * b, * c, * d;

	a = (int*)malloc(sizeof(int) * a_r * a_c);
	b = (int*)malloc(sizeof(int) * b_r * b_c);
	c = (int*)malloc(sizeof(int) * a_r * b_c);
	d = (int*)malloc(sizeof(int) * a_r * b_c);
	for (int i = 0; i < a_r * b_c; i++)
		d[i] = 0;
	srand(0);

	//�����������
	matgen(a, a_r,a_c);
	matgen(b, b_r,b_c);
	/*
	ofstream outA;
	outA.open("A.csv", ios::out);
	for (int i = 0; i < a_r * a_c; i++) {
		if (i % a_r == 0)
			outA << endl;
		outA<< a[i]<<',';
	}
	outA.close();
	ofstream outB;
	outB.open("B.csv", ios::out);
	for (int i = 0; i < b_r * b_c; i++) {
		if (i % b_r == 0)
			outB << endl;
		outB << b[i]<<',';
	}
	outB.close();
*/

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
	MatMatMul<TILE_WIDTH><<<dimGrid,dimBlock>>>(cuda_a,cuda_b,cuda_c,a_r,a_c/*b_r*/,b_c);


	cudaMemcpy(c, cuda_c, sizeof(int) * a_r * b_c, cudaMemcpyDeviceToHost);

	clock_t ep = clock();
	cout << "end of CUDA and start of CPU:" << ep << endl;
	cout << "cost:" << (double)(ep - sp) / CLOCKS_PER_SEC;
	//����������֤
	/*
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				d[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
	ep = clock();
	cout << "end of CPU:" << ep << endl;
	for (int i = 0; i < n; i++)
	{
		cout << d[i * n + i] << ' ' << c[i * n + i] << endl;
	}*/


	cudaFreeHost(a);
	cudaFreeHost(b);
	//�ͷ��Դ�ռ�
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);

	return 0;
	
}




