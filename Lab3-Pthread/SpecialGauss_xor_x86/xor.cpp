#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <windows.h>
#include<pthread.h>
#include <semaphore.h>
using namespace std;

const int maxColNum=8399;
const int elitNum=6375; //��Ԫ��
const int elitLineNum=4535; //����Ԫ��

string s1="7.1.txt";
string s2="7.2.txt";



//class BitList;

//int countNum=maxColNum-1;
//int ptr=maxColNum-elitNum-1;
//int nextArray[maxColNum-elitNum]; //�����Ԫ�ӵ�����
//int isTurned[elitLineNum]={0}; //��Ǳ���Ԫ���Ƿ�����
//int reverseNext[maxColNum];
//int nextRow[maxColNum-elitNum];
//int nextRow=-1;
//int ptr=maxColNum-elitNum-1;


//����λͼ���������������ݴ���λͼ��
class BitList
{
public:
    int byteNum; //�ֽ���
    int bitNum; //λ��
    int *byteArray;
    int pivot; //��Ԫ
public:
    BitList()
    {
        byteNum=0;
        bitNum=0;
        pivot=-1;
        byteArray=NULL;
    }
    bool isNull()
    {
        if(this->pivot==-1)
            return true;
        else
            return false;
    }
    void init(int n)
    {
        bitNum=n; //λ�ĸ�����ÿ��Ԫ�ظ�����ͬ
        byteNum=(n+31)/32;
        byteArray=new int[byteNum];
        for(int i=0;i<byteNum;i++)
        {
            byteArray[i]=0b00000000000000000000000000000000;
        }

    }
    bool insert(int col) //�������ݸ�������1������
    {
        if(col>=0 && col/32<byteNum)
        {
            int bits=col&31;
            byteArray[col/32]=(byteArray[col/32]) | (1<<bits);
            if(col>pivot)
                pivot=col;
            return true;
        }
        return false;

    }
    int getPivot()
    {
        return this->pivot;
    }
    void resetPivot()
    {
        int p=-1;
        for(int i=0;i<bitNum;i++)
        {
            if((byteArray[i/32]&(1<<(i&31))))
            {
                p=i;
            }
        }
        this->pivot=p;
    }

    //�����������
    bool xorBitList(BitList b)
    {
        if(this->bitNum!=b.bitNum)
            return false;
        for(int i=0;i<byteNum;i++)
        {
            this->byteArray[i]=this->byteArray[i]^b.byteArray[i];
        }
        return true;
    }

    //���ñ���Ԫ��Ϊ��Ԫ��
    void setBitList(BitList b)
    {
        this->bitNum=b.bitNum;
        this->byteNum=b.byteNum;
        this->pivot=b.pivot;
        for(int i=0;i<byteNum;i++)
        {
            this->byteArray[i]=b.byteArray[i];
        }
    }

    void show()
    {
        for(int i=bitNum-1;i>=0;i--)
        {
            if((byteArray[i/32])&(1<<(i&31)))
            {
                cout<<i<<" ";
            }
        }
        cout<<endl;
    }

};

BitList elit[maxColNum];
BitList elitLine[elitLineNum];


//��������
void inputData()
{
    int data;
    //elit=new BitList[maxColNum];//��Ҫ�����в�����ӱ����Ԫ�ӵı���Ԫ��
    //elitLine=new BitList[elitLineNum];
    for(int i=0;i<maxColNum;i++)
    {
        elit[i].init(maxColNum);
    }
    for(int i=0;i<elitLineNum;i++)
    {
        elitLine[i].init(maxColNum);
    }
    ifstream elitInput;
    string elitString;
    elitInput.open(s1);
    int flag1=0;
    int elitRow=0;
    while(getline(elitInput,elitString))
    {
        istringstream in(elitString);
        while(in>>data)
        {
            if(data>=0&&data<=maxColNum)
			{
				if(flag1==0)
				{
					elitRow=data; //����˳��������λ�ڶԽ���
				}
				elit[elitRow].insert(data);
				flag1=1;
			}
        }
        flag1=0;
    }
    elitInput.close();
    ifstream elitLineInput;
    string elitLineString;
    elitLineInput.open(s2);
    int count=0;
    while(getline(elitLineInput,elitLineString))
    {
        istringstream in(elitLineString);
        while(in>>data)
        {
            if(data>=0&&data<=maxColNum)
            {
                elitLine[count].insert(data);
            }
        }
        count++;
    }
    elitLineInput.close();

}



const int NUM_THREADS=7;//������߳���

//�߳����ݽṹ����
typedef struct{
    int t_id; //�߳�id
}threadParam_t;

//barrier����
pthread_barrier_t barrier_Xor;
pthread_barrier_t barrier_Xor2;
pthread_barrier_t barrier_Xor3;

//�̺߳���
void* threadFunc(void* param)
{
    threadParam_t* p=(threadParam_t*)param;
    int t_id=p->t_id;
    for(int j=0;j<elitLineNum;j++)
    {
        bool flag=elit[elitLine[j].getPivot()].isNull();
        while(!elitLine[j].isNull()&&!flag)
        {
                pthread_barrier_wait(&barrier_Xor);
                //���
                for(int i=t_id;i<elitLine[j].byteNum;i+=NUM_THREADS)
                {
                    elitLine[j].byteArray[i]=elitLine[j].byteArray[i]^elit[elitLine[j].getPivot()].byteArray[i];
                }
                pthread_barrier_wait(&barrier_Xor);
                //������Ԫ
                int p=-1;
                for(int i=0;i<elitLine[j].bitNum;i++)
                {
                    if((elitLine[j].byteArray[i/32]&(1<<(i&31))))
                        p=i;
                }
                elitLine[j].pivot=p;
                flag=elit[elitLine[j].getPivot()].isNull();
                pthread_barrier_wait(&barrier_Xor);

        }
        pthread_barrier_wait(&barrier_Xor2);
        if(t_id==0&&!elitLine[j].isNull())
        {
            //����Ϊ��Ԫ��
            elit[elitLine[j].getPivot()].bitNum=elitLine[j].bitNum;
            elit[elitLine[j].getPivot()].byteNum=elitLine[j].byteNum;
            elit[elitLine[j].getPivot()].pivot=elitLine[j].pivot;
            for(int i=0;i<elitLine[j].byteNum;i++)
            {
                elit[elitLine[j].getPivot()].byteArray[i]=elitLine[j].byteArray[i];
            }
        }
        pthread_barrier_wait(&barrier_Xor3);
    }
    pthread_exit(NULL);
}



void display(BitList* b,int n)
{
    cout<<"Result:"<<endl;
    if((n<0)|(n>maxColNum)|(b==NULL))
        return;
    else
    {
        for(int i=0;i<n;i++)
            b[i].show();
    }
}

int main()
{
    long long head, tail, freq;

    //inputNum();
    inputData();
    //storeNullElit();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    //��ʼ��barrier
    pthread_barrier_init(&barrier_Xor,NULL,NUM_THREADS);
    pthread_barrier_init(&barrier_Xor2,NULL,NUM_THREADS);
    pthread_barrier_init(&barrier_Xor3,NULL,NUM_THREADS);
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
    pthread_barrier_destroy(&barrier_Xor);
    pthread_barrier_destroy(&barrier_Xor2);
    pthread_barrier_destroy(&barrier_Xor3);

    //serialSpecialGauss();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout <<"�������: "<<maxColNum<< " ������Ԫ��: " << elitNum <<" ����Ԫ��: "<<elitLineNum<<" time: "<<(tail-head)*1000.0/freq<< "ms" << endl;

    //display(elitLine,elitLineNum);
}











