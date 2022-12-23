#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
using namespace std;

//两处均使用AVX的高斯消去（AVX是256位）
//未对齐
float** parallelGauss(float** matrix,int N)
{
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[8]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        t1 = _mm256_loadu_ps(temp); //t1中保存8个相同主元
        int j;
        for(j=k;j<=N-8;j+=8){ //不对齐直接进行
            t2=_mm256_loadu_ps(matrix[k]+j);
            t3=_mm256_div_ps(t2,t1); //一次执行4个除法运算
            _mm256_storeu_ps(matrix[k]+j,t3);//除法结果存回
        }
        //处理末尾剩余
        if(j<N){
            for(;j<N;j++){
                matrix[k][j]=matrix[k][j]/temp[0];
            }
        }

        for(int i=k+1;i<N;i++){
            float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm256_loadu_ps(temp2); //保存8个减数
            for(j=k+1;j<=N-8;j+=8){ //不考虑对齐
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
    }
    return matrix;
}

//两处优化对齐
float** parallelGaussAlign(float** matrix,int N)
{
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[8]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        long long int pt=(long long int)&temp[0];

        t1 = _mm256_set_ps(matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]); //t1中保存8个相同主元
        int align1;
        if(k%8!=0){ //先处理前面不对齐的元素
            align1=k+8-k%8;
            for(int p=k;p<align1;p++){
                matrix[k][p]=matrix[k][p]/temp[0];
            }
        }
        else{
            align1=k;
        }

        int j;
        for(j=align1;j<=N-8;j+=8){ //对齐进行
            t2=_mm256_load_ps(matrix[k]+j);
            t3=_mm256_div_ps(t2,t1); //一次执行8个除法运算
            _mm256_store_ps(matrix[k]+j,t3);//除法结果存回
        }
        //处理末尾剩余
        if(j<N){
            for(;j<N;j++){
                matrix[k][j]=matrix[k][j]/temp[0];
            }
        }

        for(int i=k+1;i<N;i++){
            float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm256_set_ps(matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]); //保存8个减数
            int align2;
            if((k+1)%8!=0){
                align2=(k+1)+8-(k+1)%8;
                for(int p=k+1;p<align2;p++){
                    matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
                }
            }
            else{
                align2=k+1;
            }

            for(j=align2;j<=N-8;j+=8){ //考虑对齐
                t2=_mm256_load_ps(matrix[i]+j);
                t3=_mm256_load_ps(matrix[k]+j);
                t4=_mm256_sub_ps(t2,_mm256_mul_ps(t1,t3));
                _mm256_store_ps(matrix[i]+j,t4);
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
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[8]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        t1 = _mm256_loadu_ps(temp); //t1中保存8个相同主元
        int j;
        for(j=k;j<=N-8;j+=8){ //不对齐直接进行
            t2=_mm256_loadu_ps(matrix[k]+j);
            t3=_mm256_div_ps(t2,t1); //一次执行4个除法运算
            _mm256_storeu_ps(matrix[k]+j,t3);//除法结果存回
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
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float temp[8]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
        long long int pt=(long long int)&temp[0];

        t1 = _mm256_set_ps(matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]); //t1中保存8个相同主元
        int align1;
        if(k%8!=0){ //先处理前面不对齐的元素
            align1=k+8-k%8;
            for(int p=k;p<align1;p++){
                matrix[k][p]=matrix[k][p]/temp[0];
            }
        }
        else{
            align1=k;
        }

        int j;
        for(j=align1;j<=N-8;j+=8){ //对齐进行
            t2=_mm256_load_ps(matrix[k]+j);
            t3=_mm256_div_ps(t2,t1); //一次执行8个除法运算
            _mm256_store_ps(matrix[k]+j,t3);//除法结果存回
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
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float  pivot = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / pivot;
        }

        for(int i=k+1;i<N;i++){
            float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm256_loadu_ps(temp2); //保存8个减数
            int j;
            for(j=k+1;j<=N-8;j+=8){ //不考虑对齐
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
    }
    return matrix;
}
//考虑对齐
float** secondParallelAlign(float** matrix,int N)
{
    __m256 t1,t2,t3,t4;
    for (int k = 0; k < N; k++)
    {
        float  pivot = matrix[k][k];
        for (int j = k; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / pivot;
        }

        for(int i=k+1;i<N;i++){
            float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            t1=_mm256_set_ps(matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]); //保存8个减数
            int align2;
            if((k+1)%8!=0){
                align2=(k+1)+8-(k+1)%8;
                for(int p=k+1;p<align2;p++){
                    matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
                }
            }
            else{
                align2=k+1;
            }
            int j;
            for(j=align2;j<=N-8;j+=8){ //考虑对齐
                t2=_mm256_load_ps(matrix[i]+j);
                t3=_mm256_load_ps(matrix[k]+j);
                t4=_mm256_sub_ps(t2,_mm256_mul_ps(t1,t3));
                _mm256_store_ps(matrix[i]+j,t4);
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
void setMatrix1(float** matrix)
{
    long long head, tail, freq;
    int N[9]={8,32,128,256,512,1024,2048,4096,8192};

    for(int p=0;p<9;p++)
    {
        matrix=new float*[N[p]];
        for (int i = 0; i < N[p]; i++)
        {
            matrix[i]  = new float[N[p]];
        }
        srand((unsigned)time(0));
        for(int i=0;i<N[p];i++){
            for(int j=0;j<i;j++){
                matrix[i][j]=0.0;
            }
            matrix[i][i]=1.0;
            for(int j=i+1;j<N[p];j++){
                matrix[i][j]=rand()%100;
            }
        }
         struct timeval start;
         struct timeval end;
         unsigned long diff;
         gettimeofday(&start,NULL);
         float** M1=parallelGauss(matrix,N[p]);
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"两处均优化+不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;

         gettimeofday(&start,NULL);
         float** M3=firstParallel(matrix,N[p]);
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"优化一处+不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;

         gettimeofday(&start,NULL);
         float** M5=secondParallel(matrix,N[p]);
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"优化二处+不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;
         cout<<endl;

    }

}

void setMatrix2(float** matrix)
{
    long long head, tail, freq;
    int N[9]={8,32,128,256,512,1024,2048,4096,8192};
    for(int p=0;p<9;p++)
    {
        matrix =reinterpret_cast<float**>(_mm_malloc(sizeof(float*)*N[p], 32));
        for (int i = 0; i < N[p]; i++)
        {
              matrix[i]  = reinterpret_cast<float*>(_mm_malloc(sizeof(float)*N[p], 32));
        }

        srand((unsigned)time(0));
        for(int i=0;i<N[p];i++){
            for(int j=0;j<i;j++){
                matrix[i][j]=0.0;
            }
            matrix[i][i]=1.0;
            for(int j=i+1;j<N[p];j++){
                matrix[i][j]=rand()%100;
            }
        }
        struct timeval start;
         struct timeval end;
         unsigned long diff;
         gettimeofday(&start,NULL);
         float** M2=parallelGaussAlign(matrix,N[p]);
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"两处均优化+不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;

         gettimeofday(&start,NULL);
         float** M4=firstParallelAlign(matrix,N[p]);
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"优化一处+不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;

         gettimeofday(&start,NULL);
         float** M6=secondParallelAlign(matrix,N[p]);
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"优化二处+不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;
         cout<<endl;
    }
}



int main()
{
    float** matrix1;
    setMatrix1(matrix1);
    float** matrix2;
    setMatrix2(matrix2);

    return 0;

}
