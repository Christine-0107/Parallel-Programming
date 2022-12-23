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
//ȫ�ֱ���
const int N=2048;
float matrix[N][N];
const int NUM_THREADS=4;//������߳���

//�߳����ݽṹ����
typedef struct{
    int t_id; //�߳�id
}threadParam_t;

//barrier����
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

//�̺߳�������
void* threadFunc(void* param){
    float32x4_t t1,t2,t3,t4;
    threadParam_t* p=(threadParam_t*)param;
    int t_id=p->t_id;
    for(int k=0;k<N;k++){
        //t_idΪ0���߳��������������ȵȴ�
        if(t_id==0){
            for(int j=k+1;j<N;j++){
                matrix[k][j]=matrix[k][j]/matrix[k][k];
            }
            matrix[k][k]=1.0;
        }

        //��һ��ͬ���㣬�ȳ�������
        pthread_barrier_wait(&barrier_Division);

        //ѭ����������
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS){
            float temp[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=vld1q_f32(temp);
            //��ȥ
            int j;
            for(j=k+1;j<N-4;j+=4){
                t2=vld1q_f32(matrix[i]+j);
                t3=vld1q_f32(matrix[k]+j);
                t4=vsubq_f32(t2,vmulq_f32(t1,t3));
                vst1q_f32(matrix[i]+j,t4);
            }
            if(j<N){
                for(;j<N;j++){
                    matrix[i][j]=matrix[i][j]-temp[0]*matrix[k][j];
                }
            }
            matrix[i][k]=0.0;
        }

        //�ڶ���ͬ���㣬����ȥ����
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
    struct  timeval start;
    struct  timeval end;
    unsigned  long diff;
    //��ʼ������
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

    //��ʼ��barrier
    pthread_barrier_init(&barrier_Division,NULL,NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination,NULL,NUM_THREADS);

    //�����߳�
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

    gettimeofday(&end, NULL);
    diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    cout <<"N: "<<N<< " time: " << diff/1000 << "ms" << endl;
    return 0;
}
