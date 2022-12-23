#include<iostream>
#include<windows.h>
#include<stdlib.h>
#include<xmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include <semaphore.h>
#include <time.h>
using namespace std;

//ȫ�ֱ���
const int N=5200;
float matrix[N][N];
const int NUM_THREADS=4;//������߳���

//�߳����ݽṹ����
typedef struct{
    int t_id; //�߳�id
}threadParam_t;

//�ź�������
sem_t sem_leader;
sem_t sem_Division[NUM_THREADS-1];
sem_t sem_Elimination[NUM_THREADS-1];

//�̺߳�������
void* threadFunc(void* param){
    __m256 t1,t2,t3,t4;
    threadParam_t* p=(threadParam_t*)param;
    int t_id=p->t_id;
    for(int k=0;k<N;k++){
        //t_idΪ0���߳����������������������߳��ȵȴ�
        //����ֻ������һ�������̸߳����������
        if(t_id==0){
            for(int j=k+1;j<N;j++){
                matrix[k][j]=matrix[k][j]/matrix[k][k];
            }
            matrix[k][k]=1.0;
        }
        else{
            sem_wait(&sem_Division[t_id-1]);//�������ȴ���ɳ�������
        }

        //t_idΪ0���̻߳������������̣߳�������ȥ����
        if(t_id==0){
            for(int i=0;i<NUM_THREADS-1;i++){
                sem_post(&sem_Division[i]);
            }
        }

        //ѭ����������
        for(int i=k+1+t_id;i<N;i+=NUM_THREADS){
            float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm256_loadu_ps(temp2); //����8������
            //��ȥ
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
            matrix[i][k]=0.0;
        }
        if(t_id==0){
            for(int i=0;i<NUM_THREADS-1;i++){
                sem_wait(&sem_leader); //�ȴ�����worker�����ȥ
            }
            for(int i=0;i<NUM_THREADS-1;i++){
                sem_post(&sem_Elimination[i]);//֪ͨ����worker������һ��
            }
        }
        else{
            sem_post(&sem_leader);//֪ͨleader���������ȥ
            sem_wait(&sem_Elimination[t_id-1]);//�ȴ�֪ͨ��������һ��
        }

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

    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    //��ʼ���ź���
    sem_init(&sem_leader,0,0);
    for(int i=0;i<NUM_THREADS-1;i++){
        sem_init(&sem_Division[i],0,0);
        sem_init(&sem_Elimination[i],0,0);
    }

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

    //�����ź���
    sem_destroy(&sem_leader);
    for(int i=0;i<NUM_THREADS;i++){
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout<<"N: "<<N<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
    return 0;
}
