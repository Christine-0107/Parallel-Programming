#include<iostream>
//#include<windows.h>
#include<stdlib.h>
//#include<xmmintrin.h>
//#include<immintrin.h>
#include<pthread.h>
#include <semaphore.h>
#include <time.h>
#include<sys/time.h>
#include<arm_neon.h>
using namespace std;

//全局变量
const int N=2048;
float matrix[N][N];
const int NUM_THREADS=4;//分配的线程数

//线程数据结构定义
typedef struct{
    int t_id;
}threadParam_t;

//信号量定义
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];//每个线程的信号量
sem_t sem_workerend[NUM_THREADS];

//线程函数定义
void* threadFunc(void* param){
    threadParam_t* p=(threadParam_t*)param;
    int t_id=p->t_id;
    for(int k=0;k<N;k++){
        sem_wait(&sem_workerstart[t_id]);//阻塞，等待主线程完成除法操作
        //循环划分任务
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS){
            //消去
            for(int j=k+1;j<N;j++){
                matrix[i][j]=matrix[i][j]-matrix[i][k]*matrix[k][j];
            }
            matrix[i][k]=0.0;
        }
        sem_post(&sem_main);//唤醒主线程
        sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮

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

int main(){
    struct  timeval start;
    struct  timeval end;
    unsigned  long diff;
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

    gettimeofday(&start, NULL);

    //初始化信号量
    sem_init(&sem_main,0,0);
    for(int i=0;i<NUM_THREADS;i++){
        sem_init(&sem_workerstart[i],0,0);
        sem_init(&sem_workerend[i],0,0);
    }

    //创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for(int t_id=0;t_id<NUM_THREADS;t_id++){
        param[t_id].t_id=t_id;
        pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
    }

    for(int k=0;k<N;k++){
        //主线程做除法操作，此时工作线程处于阻塞状态
        for(int j=k+1;j<N;j++){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
        }
        matrix[k][k]=1.0;

        //开始唤醒工作线程
        for(int t_id=0;t_id<NUM_THREADS;t_id++){
            sem_post(&sem_workerstart[t_id]);
        }
        //主线程睡眠（等待所有工作线程完成此轮消去）
        for(int t_id=0;t_id<NUM_THREADS;t_id++){
            sem_wait(&sem_main);
        }
        //主线程再次唤醒工作线程进入下一轮消去
        for(int t_id=0;t_id<NUM_THREADS;t_id++){
            sem_post(&sem_workerend[t_id]);
        }

    }
    for(int t_id=0;t_id<NUM_THREADS;t_id++){
        pthread_join(handles[t_id],NULL);
    }

    //销毁信号量
    sem_destroy(&sem_main);
    for(int i=0;i<NUM_THREADS;i++){
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }

    gettimeofday(&end, NULL);
    diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    cout <<"N: "<<N<< " time: " << diff/1000 << "ms" << endl;
    return 0;
}
