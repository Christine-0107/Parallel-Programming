#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>
using namespace std;

const int NUM_THREADS=4;

//不考虑对齐
void avxOmpGauss(float** matrix,int N)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            float temp[8]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
            __m256 t1=_mm256_loadu_ps(temp);
            int j;
            __m256 t2,t3;
            for(j=k;j<=N-8;j+=8)
            {
                t2=_mm256_loadu_ps(matrix[k]+j);
                t3=_mm256_div_ps(t2,t1);
                _mm256_storeu_ps(matrix[k]+j,t3);
            }
            if(j<N)
            {
                for(;j<N;j++)
                {
                    matrix[k][j]=matrix[k][j]/temp[0];
                }
            }
        }

        #pragma omp for
        for(int i=k+1;i<N;i++){
            float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            __m256 t1=_mm256_loadu_ps(temp2); //保存8个减数
            int j;
            __m256 t2,t3,t4;
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
    
}
//考虑对齐
void avxOmpGaussAlign(float** matrix,int N)
{
    
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            float temp[8]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
            //__m256 t1 = _mm256_set_ps(matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]); 
            __m256 t1=_mm256_load_ps(temp);
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
            __m256 t2,t3;
            for(j=align1;j<=N-8;j+=8)
            {
                t2=_mm256_load_ps(matrix[k]+j);
                t3=_mm256_div_ps(t2,t1);
                _mm256_store_ps(matrix[k]+j,t3);
            }
            if(j<N)
            {
                for(;j<N;j++)
                {
                    matrix[k][j]=matrix[k][j]/temp[0];
                }
            }
        }

        #pragma omp for
        for(int i=k+1;i<N;i++){
            float temp2[8]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            //__m256 t1=_mm256_set_ps(matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]); //保存8个减数
            __m256 t1=_mm256_load_ps(temp2);
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
            __m256 t2,t3,t4;
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
    int N[10]={8,32,128,256,512,1024,2048,3000,4096,5200};

    for(int p=0;p<10;p++)
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
         for(int count=0;count<3;count++)
         {
         avxOmpGauss(matrix,N[p]);
         }
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"avx不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;

    }

}

void setMatrix2(float** matrix)
{
    long long head, tail, freq;
    int N[10]={8,32,128,256,512,1024,2048,3000,4096,5200};
    for(int p=0;p<10;p++)
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
         for(int count=0;count<3;count++)
         {
         avxOmpGaussAlign(matrix,N[p]);
         }
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"avx考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;

    }
}



int main()
{
    float** matrix1;
    setMatrix1(matrix1);
    cout<<endl;
    float** matrix2;
    setMatrix2(matrix2);

    return 0;

}
