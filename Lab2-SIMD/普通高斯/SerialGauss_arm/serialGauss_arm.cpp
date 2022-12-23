#include <sys/time.h>
#include <iostream>
#include <time.h>
using namespace std;

//串行高斯消去
float** serialGauss(float** matrix,int N)
{
    for (int k = 0; k < N; k++)
    {
        float  pivot = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / pivot;
        }
        for (int i = k + 1; i < N; i++)
        {
            float temp = matrix[i][k];
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - temp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    return matrix;
}

//打印矩阵
void print(float** matrix,int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

//生成测试样例
void setMatrix(float** matrix,int N)
{
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
}


int main()
{
    //long long head, tail, freq;
    int N[8]={8,32,128,256,512,1024,2048,4096};
    for(int p=0;p<8;p++)
    {
        float** matrix = new float* [N[p]];
        for (int i = 0; i < N[p]; i++)
        {
            matrix[i] = new float[N[p]];
        }
        setMatrix(matrix,N[p]);
        struct  timeval start;
        struct  timeval end;
        unsigned  long diff;
        gettimeofday(&start, NULL);
        //QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        //QueryPerformanceCounter((LARGE_INTEGER *)&head);
        float** M = serialGauss(matrix,N[p]);
        //QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        gettimeofday(&end, NULL);
        diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
        cout <<"N: "<<N[p]<< " time: " << diff << "us" << endl;
        //print(M,N[p]);
        //cout<<"N: "<<N[p]<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
    }
    return 0;

}
