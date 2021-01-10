/* filename: matMultiplyWithMPI.cpp
 * parallel matrix multiplication with MPI
 * C(m,n) = A(m,p) * B(p,n)
 * input: three parameters - m, p, n
 * @copyright: fengfu-chris
 */
#include<iostream>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>
#include<ctime>
using namespace std;

void initMat(int* A, int rows, int cols);
void matMultiplyBlock(int* A, int* B, int* matResult, int m, int p, int n);
void SumC(int C[], int m, int n);
int main(int argc, char** argv)
{
    int m = atoi(argv[1]);
    int p = atoi(argv[2]);
    int n = atoi(argv[3]);

    int* A =NULL, * B = NULL, * C =NULL ,*D =NULL;
    int* blockA = NULL, * blockB = NULL;

    int myrank, numprocs;

    MPI_Status status;
    //并行计时开始
    

    MPI_Init(&argc, &argv);  // 并行开始
    //得到所有可以工作的进程数量
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    // 得到当前进程的秩
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // 得到进程的名字
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Msg from processor %s, rank %d out of %d processors\n",
        processor_name, myrank, numprocs);
    int bloce_size = m / numprocs;

    blockA = new int[bloce_size * p];
    B = new int[p * n];
    blockB = new int[bloce_size * n];
    clock_t p_start, p_end;
    if (myrank == 0) {
        p_start = clock();
        
        A = new int[m * p];
        C = new int[m * n];
        D = new int[m * n];

        initMat(A, m, p);
        initMat(B, p, n);
        cout << "data generation done! \n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

   
    /* step 1: 数据分配 */
    //https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/zh_cn/
    //当根节点(在我们的例子是节点0)调用 MPI_Bcast 函数的时候，data 变量里的值会被发送到其他的节点上。
    //当其他的节点调用 MPI_Bcast 的时候，data 变量会被赋值成从根节点接受到的数据。
    //https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/zh_cn/
    MPI_Scatter(A, bloce_size * p, MPI_INT, blockA, bloce_size * p, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, p * n, MPI_INT, 0, MPI_COMM_WORLD);

    /* step 2: 并行计算C的各个分块 */
    matMultiplyBlock(blockA, B, blockB, bloce_size, p, n);

    MPI_Barrier(MPI_COMM_WORLD);

    /* step 3: 汇总结果 */
    MPI_Gather(blockB, bloce_size * n, MPI_INT, C, bloce_size * n, MPI_INT, 0, MPI_COMM_WORLD);

    /* step 3-1: 解决历史遗留问题（多余的分块） */
    int remainRowsStartId = bloce_size * numprocs;
    if (myrank == 0 && remainRowsStartId < m) {
        int remainRows = m - remainRowsStartId;
        matMultiplyBlock(A + remainRowsStartId * p, B, C + remainRowsStartId * n, remainRows, p, n);
    }

    
    MPI_Finalize(); // 并行结束
    if (myrank == 0) {
        SumC(C, m, n);
        //PrintC(C, m, n);
        p_end = clock();
        cout << "Parallel Computation costs : " << p_end - p_start << endl;
        

        //串行计算
        //CPU矩阵乘法，存入矩阵D
        /*
        clock_t s_start = clock(), s_end;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int temp = 0;
                for (int k = 0; k < p; k++) {
                    temp += A[i * p + k] * B[k * n + j];
                }
                D[i * n + j] = temp;
            }
        }
        s_end = clock();
        SumC(D, m, n);
        cout << "Serial Computation costs : " << s_end - s_start << endl;*/

        delete[] A;
        delete[] C;
    }

    delete[] blockA;
    delete[] B;
    delete[] blockB;

   

    

    return 0;
}

void SumC(int C[], int m, int n) {
    long int res = 0;
    for (int i = 0; i < m * n; i++) {
        res += C[i];
    }
    cout << "Parallel final sum = " << res<<endl;
}

void PrintC(int C[],int m,int n) {
    for (int i = 0; i < m * n; i++) {
        if (i % m == 0) {
            std::cout << std::endl;
        }
        std::cout << C[i] << ' ';
    }
}

void initMat(int* A, int rows, int cols)
{
    srand(0);
    for (int i = 0; i < rows * cols; i++) {
        A[i] = (int)rand() %10;
    }
}
void matMultiplyBlock(int* A, int* B, int* matResult, int m, int p, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int temp = 0;
            for (int k = 0; k < p; k++) {
                temp += A[i * p + k] * B[k * n + j];
            }
            matResult[i * n + j] = temp;
        }
    }
}