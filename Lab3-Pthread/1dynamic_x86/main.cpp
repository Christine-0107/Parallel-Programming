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
const int N=2048;
float matrix[N][N];
const int NUM_THREADS=4;//������߳���
const int NUM_ROWS=N/NUM_THREADS; //ÿ���̸߳�����еĸ���

typedef struct{
    int k; //��ȥ���ִ�
    int t_id; //�߳�id
}threadParam_t;

void* threadFunc(void* param){
    threadParam_t* p=(threadParam_t*)param;
    int k=p->k; //��ȥ���ִ�
    int t_id=p->t_id; //�̱߳��
    //int i=k+t_id+1; //��ȡ�Լ��ļ�������
    for(int i=k+t_id*NUM_ROWS+1;i<min(k+t_id*NUM_ROWS+1+NUM_ROWS,N);i++){
        for(int j=k+1;j<N;j++){
            matrix[i][j]=matrix[i][j]-matrix[i][k]*matrix[k][j];
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
    for(int k=0;k<N;k++){
        //���߳�����������
        for(int j=k+1;j<N;j++){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
        }
        matrix[k][k]=1.0;

        //���������̣߳�������ȥ����
        int worker_count=NUM_THREADS; //�����̵߳�����
        pthread_t* handles=new pthread_t[worker_count]; //������Ӧ��Handle����̬����
        threadParam_t* param=new threadParam_t[worker_count]; //������Ӧ���߳����ݽṹ����̬����

        //��������
        for(int t_id=0;t_id<worker_count;t_id++){
            param[t_id].k=k;
            param[t_id].t_id=t_id;
        }

        //�����߳�
        for(int t_id=0;t_id<worker_count;t_id++){
            pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
        }

        //���̹߳�������й����߳���ɴ�����ȥ����
        for(int t_id=0;t_id<worker_count;t_id++){
            pthread_join(handles[t_id],NULL);
        }

    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout<<"N: "<<N<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
    return 0;
}
