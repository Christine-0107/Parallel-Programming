#include<iostream>
#include<windows.h>
#include<stdlib.h>
#include<xmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include <semaphore.h>
#include <time.h>
using namespace std;

//全局变量
const int N=8;
float matrix[N][N];
const int NUM_THREADS=4;//分配的线程数
const int NUM_ROWS=N/NUM_THREADS; //每个线程负责的行的个数

typedef struct{
    int k; //消去的轮次
    int t_id; //线程id
}threadParam_t;

void* threadFunc(void* param){
    threadParam_t* p=(threadParam_t*)param;
    int k=p->k; //消去的轮次
    int t_id=p->t_id; //线程编号
    //int i=k+t_id+1; //获取自己的计算任务
    __m256 t1,t2,t3,t4;
    for(int i=k+t_id*NUM_ROWS+1;i<min(k+t_id*NUM_ROWS+1+NUM_ROWS,N);i++){
        float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
        t1=_mm256_loadu_ps(temp2); //保存8个减数
        int j;
        for(j=k+1;j<N-8;j+=8){
            t2=_mm256_loadu_ps(matrix[i]+j);
            t3=_mm256_loadu_ps(matrix[k]+j);
            t4=_mm256_sub_ps(t2,_mm256_mul_ps(t1,t3));
            _mm256_storeu_ps(matrix[i]+j,t4);
        }
        if(j<N){
            for(;j<N;j++){
                matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
            }
        }
        matrix[i][k]=0;
    }
    pthread_exit(NULL);
}

void display(){
	for(int i = 0; i < N; i ++){
		for(int j = 0; j < N; j ++){
			cout<<matrix[i][j]<<" ";
		}
		cout<<endl;
	}
}



int main()
{
    long long head, tail, freq;
    //初始化数组
    srand((unsigned)time(0));
    for(int i=0;i<N;i++){
        for(int j=0;j<i;j++){
            matrix[i][j]=0.0;
        }
        matrix[i][i]=1.0;
        for(int j=i+1;j<N;j++){
            matrix[i][j]=rand()%100;
        }
    }

    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int k=0;k<N;k++){
        //主线程做除法操作
        for(int j=k+1;j<N;j++){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
        }
        matrix[k][k]=1.0;

        //创建工作线程，进行消去操作
        int worker_count=NUM_THREADS; //工作线程的数量
        pthread_t* handles=new pthread_t[worker_count]; //创建对应的Handle，动态分配
        threadParam_t* param=new threadParam_t[worker_count]; //创建对应的线程数据结构，动态分配

        //分配任务
        for(int t_id=0;t_id<worker_count;t_id++){
            param[t_id].k=k;
            param[t_id].t_id=t_id;
        }

        //创建线程
        for(int t_id=0;t_id<worker_count;t_id++){
            pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
        }

        //主线程挂起等所有工作线程完成此轮消去工作
        for(int t_id=0;t_id<worker_count;t_id++){
            pthread_join(handles[t_id],NULL);
        }

    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout<<"N: "<<N<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
    return 0;
}
