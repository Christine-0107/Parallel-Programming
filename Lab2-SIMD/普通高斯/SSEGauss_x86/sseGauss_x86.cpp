//#include <sys/time.h>
#include <iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
using namespace std;

//两处均使用SSE的高斯消去（未对齐）
float** parallelGauss(float** matrix,int N)
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_loadu_ps(temp); //t1中保存4个相同主元
        int j;
        for(j=k;j<=N-4;j+=4){ //不对齐直接进行
            t2=_mm_loadu_ps(matrix[k]+j);
            t3=_mm_div_ps(t2,t1); //一次执行4个除法运算
            _mm_storeu_ps(matrix[k]+j,t3);//除法结果存回
        }
        //处理末尾剩余
        if(j<N){
            for(;j<N;j++){
                matrix[k][j]=matrix[k][j]/temp[0];
            }
        }

        for(int i=k+1;i<N;i++){
            float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_loadu_ps(temp2); //保存4个减数
            for(j=k+1;j<=N-4;j+=4){ //不考虑对齐
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
            matrix[i][k]=0;

        }
    }
    return matrix;
}

//两处优化对齐
float** parallelGaussAlign(float** matrix,int N)
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_load_ps(temp); //t1中保存4个相同主元
        if(k%4!=0){ //先处理前面不对齐的元素
            for(int p=k;p<k+4-k%4;p++){
                matrix[k][p]=matrix[k][p]/temp[0];
            }
        }
        int j;
        for(j=k+4-k%4;j<=N-4;j+=4){ //对齐进行
            t2=_mm_load_ps(matrix[k]+j);
            t3=_mm_div_ps(t2,t1); //一次执行4个除法运算
            _mm_store_ps(matrix[k]+j,t3);//除法结果存回
        }
        //处理末尾剩余
        if(j<N){
            for(;j<N;j++){
                matrix[k][j]=matrix[k][j]/temp[0];
            }
        }

        for(int i=k+1;i<N;i++){
            float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_load_ps(temp2); //保存4个减数
            if((k+1)%4!=0){
                for(int p=k+1;p<(k+1)+4-(k+1)%4;p++){
                    matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
                }
            }
            for(j=(k+1)+4-(k+1)%4;j<=N-4;j+=4){ //考虑对齐
                t2=_mm_load_ps(matrix[i]+j);
                t3=_mm_load_ps(matrix[k]+j);
                t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                _mm_store_ps(matrix[i]+j,t4);
            }
            if(j<N){
                for(;j<N;j++){
                    matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
                }
            }
            matrix[i][k]=0;

        }
    }
    return matrix;
}

//只优化第一处
//不考虑对齐
float** firstParallel(float** matrix,int N)
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_loadu_ps(temp); //t1中保存4个相同主元
        int j;
        for(j=k;j<=N-4;j+=4){ //不对齐直接进行
            t2=_mm_loadu_ps(matrix[k]+j);
            t3=_mm_div_ps(t2,t1); //一次执行4个除法运算
            _mm_storeu_ps(matrix[k]+j,t3);//除法结果存回
        }
        //处理末尾剩余
        if(j<N){
            for(;j<N;j++){
                matrix[k][j]=matrix[k][j]/temp[0];
            }
        }

        for(int i=k+1;i<N;i++){
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
//考虑对齐
float** firstParallelAlign(float** matrix,int N)
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[4]={matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm_load_ps(temp); //t1中保存4个相同主元
        if((k)%4!=0){ //先处理前面不对齐的元素
            for(int p=k;p<k+4-(k)%4;p++){
                matrix[k][p]=matrix[k][p]/temp[0];
            }
        }
        int j;
        for(j=k+4-(k)%4;j<=N-4;j+=4){ //对齐进行
            t2=_mm_load_ps(matrix[k]+j);
            t3=_mm_div_ps(t2,t1); //一次执行4个除法运算
            _mm_store_ps(matrix[k]+j,t3);//除法结果存回
        }
        //处理末尾剩余
        if(j<N){
            for(;j<N;j++){
                matrix[k][j]=matrix[k][j]/temp[0];
            }
        }

        for(int i=k+1;i<N;i++){
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

//只优化第二处
//不考虑对齐
float** secondParallel(float** matrix,int N)
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float  pivot = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / pivot;
        }

        for(int i=k+1;i<N;i++){
            float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_loadu_ps(temp2); //保存4个减数
            int j;
            for(j=k+1;j<=N-4;j+=4){ //不考虑对齐
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
            matrix[i][k]=0;

        }
    }
    return matrix;
}
//考虑对齐
float** secondParallelAlign(float** matrix,int N)
{
    __m128 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float  pivot = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / pivot;
        }

        for(int i=k+1;i<N;i++){
            float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm_load_ps(temp2); //保存4个减数
            if((k+1)%4!=0){
                for(int p=k+1;p<(k+1)+4-(k+1)%4;p++){
                    matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
                }
            }
            int j;
            for(j=(k+1)+4-(k+1)%4;j<=N-4;j+=4){ //考虑对齐
                t2=_mm_load_ps(matrix[i]+j);
                t3=_mm_load_ps(matrix[k]+j);
                t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
                _mm_store_ps(matrix[i]+j,t4);
            }
            if(j<N){
                for(;j<N;j++){
                    matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
                }
            }
            matrix[i][k]=0;

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
    long long head, tail, freq;
    int N[8]={8,32,128,256,512,1024,2048,4096};
    for(int p=0;p<8;p++)
    {
        float** matrix = new float* [N[p]];
        for (int i = 0; i < N[p]; i++)
        {
            matrix[i] = new float[N[p]];
        }
        setMatrix(matrix,N[p]);
        // struct  timeval start;
         //struct  timeval end;
         //unsigned  long diff;
         //gettimeofday(&start, NULL);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        float** M1 = parallelGauss(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        // gettimeofday(&end, NULL);
         //diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
         //cout << "time is " << diff << "us" << endl;
        //print(M1,N[p]);
        cout<<"不考虑对齐："<<endl;
        cout<<"N: "<<N[p]<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;

        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        float** M2 = parallelGaussAlign(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"考虑对齐："<<endl;
        cout<<"N: "<<N[p]<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
//
//        QueryPerformanceCounter((LARGE_INTEGER *)&head);
//        float** M3 = firstParallel(matrix,N[p]);
//        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//        cout<<"只优化第一处循环+不考虑对齐："<<endl;
//        cout<<"N: "<<N[p]<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
//
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        float** M4 = firstParallelAlign(matrix,N[p]);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"只优化第一处循环+考虑对齐："<<endl;
        cout<<"N: "<<N[p]<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
//
//        QueryPerformanceCounter((LARGE_INTEGER *)&head);
//        float** M5 = secondParallel(matrix,N[p]);
//        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//        cout<<"只优化第二处循环+不考虑对齐："<<endl;
//        cout<<"N: "<<N[p]<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
//
//        QueryPerformanceCounter((LARGE_INTEGER *)&head);
//        float** M6 = secondParallelAlign(matrix,N[p]);
//        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
//        cout<<"只优化第二处循环+考虑对齐："<<endl;
//        cout<<"N: "<<N[p]<<" time: "<<(tail-head)*1000.0 / freq<<"ms"<<endl;
        cout<<endl;

//        if(p==0||p==1){
//            print(M5,N[p]);
//        }
    }
    return 0;

}
