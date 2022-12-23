#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <mpi.h>
#include <pmmintrin.h>
#include <cmath>
using namespace std;
int NUM_THREADS=4;


void setMatrix(float** matrix,int n)
{
    srand((unsigned)time(0));
    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++){
            matrix[i][j]=0.0;
        }
        matrix[i][i]=1.0;
        for(int j=i+1;j<n;j++){
            matrix[i][j]=rand()%100;
        }
    }
}

void serialGauss(float** matrix,int n)
{
    setMatrix(matrix,n);
    struct timeval start;
    struct timeval end;
    unsigned long diff;
    gettimeofday(&start, NULL);
    for (int k = 0; k < n; k++)
    {
        float  pivot = matrix[k][k];
        for (int j = k; j < n; j++)
        {
            matrix[k][j] = matrix[k][j] / pivot;
        }
        for (int i = k + 1; i < n; i++)
        {
            float temp = matrix[i][k];
            for (int j = k + 1; j < n; j++)
            {
                matrix[i][j] = matrix[i][j] - temp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    gettimeofday(&end, NULL);
    diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
    cout<<"串行"<<endl;
    cout<<"N: "<<n<<" time: "<<diff<<"us"<<endl;
}


void block_gauss(float** matrix,int my_rank, int num_proc,int n)
{
    int block_size = n / num_proc;
    int remain = n % num_proc;

    int my_begin = my_rank * block_size;
    int my_end = my_begin + block_size;
    if(my_rank==num_proc-1){
        my_end=my_begin+block_size+remain;
    }
    for (int k = 0; k < n; k++) {
        if (k >= my_begin && k < my_end) {
            float pivot = matrix[k][k];
            for (int j = k + 1; j < n; j++)
                matrix[k][j] = matrix[k][j] / pivot;
            matrix[k][k] = 1.0;
            for (int p = my_rank + 1; p < num_proc; p++)
                MPI_Send(matrix[k], n, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else {
            int current_work_p = k / block_size;
            if (current_work_p < my_rank)
                MPI_Recv(matrix[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = my_begin; i < my_end; i++) {
            if (i > k) {
                for (int j = k + 1; j < n; j++){
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0.0;
            }
        }
    }
}
void block_gauss_sse_omp(float** matrix,int my_rank, int num_proc,int n)
{
    int block_size = n / num_proc;
    int remain = n % num_proc;

    int my_begin = my_rank * block_size;
    int my_end = my_begin + block_size;
    if(my_rank==num_proc-1){
        my_end=my_begin+block_size+remain;
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        #pragma omp single
        {
            if (k >= my_begin && k < my_end){
                float temp[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
                __m128 t1=_mm_loadu_ps(temp);
                int j;
                __m128 t2,t3;
                for(j=k;j<=n-4;j+=4)
                {
                    t2=_mm_loadu_ps(matrix[k]+j);
                    t3=_mm_div_ps(t2,t1);
                    _mm_storeu_ps(matrix[k]+j,t3);
                }
                if(j<n)
                {
                    for(;j<n;j++)
                    {
                        matrix[k][j]=matrix[k][j]/temp[0];
                    }
                }
                for (int p = my_rank + 1; p < num_proc; p++)
                    MPI_Send(matrix[k], n, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }
            else {
                int current_work_p = k / block_size;
                if (current_work_p < my_rank)
                    MPI_Recv(matrix[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        #pragma omp for
        for (int i = my_begin; i < my_end; i++) {
            if (i > k) {
                float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
                __m128 t1=_mm_loadu_ps(temp2);
                int j;
                __m128 t2,t3,t4;
                for (j = k + 1; j <= n-4; j+=4){
                    t2=_mm_loadu_ps(matrix[i]+j);
                    t3=_mm_loadu_ps(matrix[k]+j);
                    t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                    _mm_storeu_ps(matrix[i]+j,t4);
                }
                if(j<n)
                {
                    for(;j<n;j++)
                    {
                        matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
                    }
                }
                matrix[i][k] = 0.0;
            }
        }
    }
}

void block_run(int version,float** matrix,int n)
{
    void (*f)(float**,int,int,int);
    string inform;
	if (version == 0) {
		f = &block_gauss;
		inform="block normal:";
	}
	else if (version == 1) {
		f = &block_gauss_sse_omp;
		inform="block+sse+omp:";
	}
	struct timeval start;
    struct timeval end;
    unsigned long diff;

    int num_proc; //进程数
    int my_rank;//正在调用的进程号
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int block_size = n / num_proc;
    int remain = n % num_proc;

    if (my_rank == 0) {
		setMatrix(matrix,n);
        gettimeofday(&start, NULL);
        for (int i = 1; i < num_proc; i++) {
            int upper_bound=block_size; //分配块数
            if(i==num_proc-1){
                upper_bound=block_size+remain;
            }
            for (int j = 0; j < upper_bound; j++)
                MPI_Send(matrix[i * block_size + j], n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        f(matrix,my_rank, num_proc,n);
        for (int i = 1; i < num_proc; i++) {
            int upper_bound=block_size;
            if(i==num_proc-1){
                upper_bound=block_size+remain;
            }
            for (int j = 0; j < upper_bound; j++)
                MPI_Recv(matrix[i * block_size + j], n, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gettimeofday(&end, NULL);
        diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
        cout<<inform<<endl;
        cout<<"N: "<<n<<" time: "<<diff<<"us"<<endl;
    }
    else {
        int upper_bound = block_size;
        if(my_rank==num_proc-1){
            upper_bound=block_size+remain;
        }
        for (int j = 0; j < upper_bound; j++)
            MPI_Recv(matrix[my_rank * block_size + j], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f(matrix,my_rank, num_proc,n);
        for (int j = 0; j < upper_bound; j++)
            MPI_Send(matrix[my_rank * block_size + j], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}

void recycle_gauss(float** matrix,int my_rank, int num_proc,int n)
{
    for (int k = 0; k < n; k++) {
        if (k % num_proc == my_rank) {
            float pivot = matrix[k][k];
            for (int j = k + 1; j < n; j++)
                matrix[k][j] = matrix[k][j] / pivot;
            matrix[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
				if (j != my_rank)
                	MPI_Send(matrix[k], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
        }
        else {
            int current_work_p = k % num_proc;
            MPI_Recv(matrix[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = my_rank; i < n; i += num_proc) {
            if (i > k) {
                for (int j = k + 1; j < n; j++){
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0.0;
            }
        }
    }
}
void recycle_gauss_sse_omp(float** matrix,int my_rank, int num_proc,int n)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        #pragma omp single
        {
            if (k % num_proc == my_rank){
                float temp[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
                __m128 t1=_mm_loadu_ps(temp);
                int j;
                __m128 t2,t3;
                for(j=k;j<=n-4;j+=4)
                {
                    t2=_mm_loadu_ps(matrix[k]+j);
                    t3=_mm_div_ps(t2,t1);
                    _mm_storeu_ps(matrix[k]+j,t3);
                }
                if(j<n)
                {
                    for(;j<n;j++)
                    {
                        matrix[k][j]=matrix[k][j]/temp[0];
                    }
                }
                for (j = 0; j < num_proc; j++)
					if (j != my_rank)
        	        	MPI_Send(matrix[k], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
            else {
                int current_work_p = k % num_proc;
                MPI_Recv(matrix[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        #pragma omp for
        for (int i = my_rank; i < n; i+=num_proc) {
            if (i > k) {
                float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
                __m128 t1=_mm_loadu_ps(temp2);
                int j;
                __m128 t2,t3,t4;
                for (j = k + 1; j <= n-4; j+=4){
                    t2=_mm_loadu_ps(matrix[i]+j);
                    t3=_mm_loadu_ps(matrix[k]+j);
                    t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                    _mm_storeu_ps(matrix[i]+j,t4);
                }
                if(j<n)
                {
                    for(;j<n;j++)
                    {
                        matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
                    }
                }
                matrix[i][k] = 0.0;
            }
        }
    }
}
void recycle_pipeline(float** matrix,int my_rank, int num_proc,int n)
{
    int pre_rank = (my_rank - 1 + num_proc) % num_proc;
	int nex_rank = (my_rank + 1) % num_proc;
    for (int k = 0; k < n; k++) {
        if (k % num_proc == my_rank) {
            float pivot = matrix[k][k];
            for (int j = k + 1; j < n; j++)
                matrix[k][j] = matrix[k][j] / pivot;
            matrix[k][k] = 1.0;
			if (nex_rank != my_rank)
	            MPI_Send(matrix[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
        }
        else {
            MPI_Recv(matrix[k], n, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (k % num_proc != nex_rank)
				MPI_Send(matrix[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
        }
        for (int i = my_rank; i < n; i += num_proc) {
            if (i > k) {
                for (int j = k + 1; j < n; j++){
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0.0;
            }
        }
    }
}
void recycle_pipeline_sse_omp(float** matrix,int my_rank, int num_proc,int n)
{
    int pre_rank = (my_rank - 1 + num_proc) % num_proc;
	int nex_rank = (my_rank + 1) % num_proc;
	#pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++) {
        #pragma omp single
        {
            if (k % num_proc == my_rank){
                float temp[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
                __m128 t1=_mm_loadu_ps(temp);
                int j;
                __m128 t2,t3;
                for(j=k;j<=n-4;j+=4)
                {
                    t2=_mm_loadu_ps(matrix[k]+j);
                    t3=_mm_div_ps(t2,t1);
                    _mm_storeu_ps(matrix[k]+j,t3);
                }
                if(j<n)
                {
                    for(;j<n;j++)
                    {
                        matrix[k][j]=matrix[k][j]/temp[0];
                    }
                }
                if (nex_rank != my_rank)
        	    	MPI_Send(matrix[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
            }
            else {
                MPI_Recv(matrix[k], n, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (nex_rank != k % num_proc)
					MPI_Send(matrix[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
            }
        }

        #pragma omp for
        for (int i = my_rank; i < n; i+=num_proc) {
            if (i > k) {
                float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
                __m128 t1=_mm_loadu_ps(temp2);
                int j;
                __m128 t2,t3,t4;
                for (j = k + 1; j <= n-4; j+=4){
                    t2=_mm_loadu_ps(matrix[i]+j);
                    t3=_mm_loadu_ps(matrix[k]+j);
                    t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                    _mm_storeu_ps(matrix[i]+j,t4);
                }
                if(j<n)
                {
                    for(;j<n;j++)
                    {
                        matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
                    }
                }
                matrix[i][k] = 0.0;
            }
        }
    }
}

void recycle_run(int version,float** matrix,int n)
{
    void (*f)(float**,int,int,int);
    string inform;
	if (version == 0) {
		f = &recycle_gauss;
		inform="循环划分普通:";
	}
	else if (version == 1) {
		f = &recycle_gauss_sse_omp;
		inform="循环划分+sse+omp:";
	}
	else if (version == 2) {
		f = &recycle_pipeline;
		inform="循环+流水线:";
	}
	else if (version == 3) {
		f = &recycle_pipeline_sse_omp;
		inform="循环+流水线+sse+omp:";
	}
	struct timeval start;
    struct timeval end;
    unsigned long diff;

    int num_proc; //进程数
    int my_rank;//正在调用的进程号
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
		setMatrix(matrix,n);
        gettimeofday(&start, NULL);
		for (int i = 0; i < n; i++) {
			int pro_row = i % num_proc;
			if (pro_row != my_rank)
				MPI_Send(matrix[i], n, MPI_FLOAT, pro_row, 0, MPI_COMM_WORLD);
		}
        f(matrix,my_rank, num_proc,n);
		for (int i = 0; i < n; i++) {
			int pro_row = i % num_proc;
			if (pro_row != my_rank)
				MPI_Recv(matrix[i], n, MPI_FLOAT, pro_row, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
        gettimeofday(&end, NULL);
        diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
        cout<<inform<<endl;
        cout<<"N: "<<n<<" time: "<<diff<<"us"<<endl;
    }
    else {
        for (int j = my_rank; j < n; j += num_proc)
            MPI_Recv(matrix[j], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f(matrix,my_rank, num_proc,n);
        for (int j = my_rank; j < n; j += num_proc)
            MPI_Send(matrix[j], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }

}

int main()
{
    int N[10]={8,32,128,256,512,1024,2048,3000,4096,5200};
    MPI_Init(NULL,NULL);
    for(int p=0;p<10;p++)
    {

        float** matrix = new float* [N[p]];
        for (int i = 0; i < N[p]; i++)
        {
            matrix[i] = new float[N[p]];
        }
        serialGauss(matrix,N[p]); //串行
        block_run(0,matrix,N[p]); //按块划分普通
        block_run(1,matrix,N[p]); //按块划分添加sse和omp
        recycle_run(0,matrix,N[p]); //循环划分普通
        recycle_run(1,matrix,N[p]);//循环划分+sse+omp
        recycle_run(2,matrix,N[p]); //流水线
        recycle_run(3,matrix,N[p]); //流水线+sse+omp


    }
    MPI_Finalize();
    return 0;

}


