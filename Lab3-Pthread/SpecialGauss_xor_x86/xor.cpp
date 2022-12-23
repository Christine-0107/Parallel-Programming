#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <windows.h>
#include<pthread.h>
#include <semaphore.h>
using namespace std;

const int maxColNum=8399;
const int elitNum=6375; //消元子
const int elitLineNum=4535; //被消元行

string s1="7.1.txt";
string s2="7.2.txt";



//class BitList;

//int countNum=maxColNum-1;
//int ptr=maxColNum-elitNum-1;
//int nextArray[maxColNum-elitNum]; //存空消元子的行数
//int isTurned[elitLineNum]={0}; //标记被消元行是否升格
//int reverseNext[maxColNum];
//int nextRow[maxColNum-elitNum];
//int nextRow=-1;
//int ptr=maxColNum-elitNum-1;


//创建位图来将倒排链表数据存入位图中
class BitList
{
public:
    int byteNum; //字节数
    int bitNum; //位数
    int *byteArray;
    int pivot; //首元
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
        bitNum=n; //位的个数与每行元素个数相同
        byteNum=(n+31)/32;
        byteArray=new int[byteNum];
        for(int i=0;i<byteNum;i++)
        {
            byteArray[i]=0b00000000000000000000000000000000;
        }

    }
    bool insert(int col) //插入数据给出的是1所在列
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

    //进行异或运算
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

    //设置被消元行为消元子
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


//输入数据
void inputData()
{
    int data;
    //elit=new BitList[maxColNum];//需要向其中不断添加变成消元子的被消元行
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
					elitRow=data; //调整顺序让首项位于对角线
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



const int NUM_THREADS=7;//分配的线程数

//线程数据结构定义
typedef struct{
    int t_id; //线程id
}threadParam_t;

//barrier定义
pthread_barrier_t barrier_Xor;
pthread_barrier_t barrier_Xor2;
pthread_barrier_t barrier_Xor3;

//线程函数
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
                //异或
                for(int i=t_id;i<elitLine[j].byteNum;i+=NUM_THREADS)
                {
                    elitLine[j].byteArray[i]=elitLine[j].byteArray[i]^elit[elitLine[j].getPivot()].byteArray[i];
                }
                pthread_barrier_wait(&barrier_Xor);
                //重置首元
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
            //升格为消元子
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

    //初始化barrier
    pthread_barrier_init(&barrier_Xor,NULL,NUM_THREADS);
    pthread_barrier_init(&barrier_Xor2,NULL,NUM_THREADS);
    pthread_barrier_init(&barrier_Xor3,NULL,NUM_THREADS);
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
    pthread_barrier_destroy(&barrier_Xor);
    pthread_barrier_destroy(&barrier_Xor2);
    pthread_barrier_destroy(&barrier_Xor3);

    //serialSpecialGauss();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout <<"最大列数: "<<maxColNum<< " 非零消元子: " << elitNum <<" 被消元行: "<<elitLineNum<<" time: "<<(tail-head)*1000.0/freq<< "ms" << endl;

    //display(elitLine,elitLineNum);
}











