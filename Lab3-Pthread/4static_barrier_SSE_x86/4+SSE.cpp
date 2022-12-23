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
const int N=2048;
float matrix[N][N];
const int NUM_THREADS=4;//分配的线程数

//线程数据结构定义
typedef struct{
    int t_id; //线程id
}threadParam_t;

//barrier定义
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

//线程函数定义
void* threadFunc(void* param){
    threadParam_t* p=(threadParam_t*)param;
    int t_id=p->t_id;
    __m128 t1,t2,t3,t4;
    for(int k=0;k<N;k++){
        //t_id为0的线程做除法，其他先等待
        if(t_id==0){
            for(int j=k+1;j<N;j++){
                matrix[k][j]=matrix[k][j]/matrix[k][k];
            }
            matrix[k][k]=1.0;
        }

        //第一个同步点，等除法做完
        pthread_barrier_wait(&barrier_Division);

        //循环划分任务
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS){
            float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_loadu_ps(temp2); //保存4个减数
            //消去
            int j;
            for(j=k+1;j<N-4;j+=4){
                t2=_mm_loadu_ps(matrix[i]+j);
                t3=_mm_loadu_ps(matrix[k]+j);
                t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                _mm_storeu_ps(matrix[i]+j,t4);
            }
            if(j<N){
                for(;j<N;j++){
                    matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
                }
            }
            matrix[i][k]=0.0;
        }

        //第二个同步点，等消去做完
        pthread_barrier_wait(&barrier_Elimination);
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

    //初始化barrier
    pthread_barrier_init(&barrier_Division,NULL,NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination,NULL,NUM_THREADS);

    //创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for(int t_id=0;t_id<NUM_THREADS;t_id++){
        param[t_id].t_id=t_id;
        pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
    }
    for(int t_id=0;t_id<NUM_THREADS;t_id++){
        pthread_join(handles[t_id],NULL);
    }
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);

    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout<<"N: "<<N<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
    return 0;
}

