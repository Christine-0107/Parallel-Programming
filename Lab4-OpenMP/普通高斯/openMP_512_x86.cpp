#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>
using namespace std;

const int NUM_THREADS=4;

//未对齐
void avx512OmpGauss(float** matrix,int N)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            float temp[16]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],
                        matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
            __m512 t1 = _mm512_loadu_ps(temp); //t1中保存16个相同主元
            int j;
            __m512 t2,t3;
            for(j=k;j<=N-16;j+=16){ //不对齐直接进行
                t2=_mm512_loadu_ps(matrix[k]+j);
                t3=_mm512_div_ps(t2,t1); //一次执行4个除法运算
                _mm512_storeu_ps(matrix[k]+j,t3);//除法结果存回
            }
            //处理末尾剩余
            if(j<N){
                for(;j<N;j++){
                    matrix[k][j]=matrix[k][j]/temp[0];
                }
            }
        }
        
        #pragma omp for
        for(int i=k+1;i<N;i++){
            float temp2[16]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],
                             matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            __m512 t1=_mm512_loadu_ps(temp2); //保存16个减数
            int j;
            __m512 t2,t3,t4;
            for(j=k+1;j<=N-16;j+=16){ //不考虑对齐
                t2=_mm512_loadu_ps(matrix[i]+j);
                t3=_mm512_loadu_ps(matrix[k]+j);
                t4=_mm512_sub_ps(t2,_mm512_mul_ps(t1,t3));
                _mm512_storeu_ps(matrix[i]+j,t4);
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

//对齐
void avx512OmpGaussAlign(float** matrix,int N)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            float temp[16]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],
                            matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
            //__m512 t1=_mm512_load_ps(temp);
            __m512 t1 = _mm512_set_ps(matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]); 
            int align1;
            if(k%16!=0){ //先处理前面不对齐的元素
                align1=k+16-k%16;
                for(int p=k;p<align1;p++){
                    matrix[k][p]=matrix[k][p]/temp[0];
                }
            }
            else{
                align1=k;
            }

            int j;
            __m512 t2,t3;
            for(j=align1;j<=N-16;j+=16){ //对齐进行
                t2=_mm512_load_ps(matrix[k]+j);
                t3=_mm512_div_ps(t2,t1); //一次执行16个除法运算
                _mm512_store_ps(matrix[k]+j,t3);//除法结果存回
            }
            //处理末尾剩余
            if(j<N){
                for(;j<N;j++){
                    matrix[k][j]=matrix[k][j]/temp[0];
                }
            }
        }

        #pragma omp for
        for(int i=k+1;i<N;i++){
            float temp2[16]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],
                             matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
            //__m512 t1=_mm512_load_ps(temp2);
            __m512 t1=_mm512_set_ps(matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]);
            int align2;
            if((k+1)%16!=0){
                align2=(k+1)+16-(k+1)%16;
                for(int p=k+1;p<align2;p++){
                    matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
                }
            }
            else{
                align2=k+1;
            }
            int j;
            __m512 t2,t3,t4;

            for(j=align2;j<=N-16;j+=16){ //考虑对齐
                t2=_mm512_load_ps(matrix[i]+j);
                t3=_mm512_load_ps(matrix[k]+j);
                t4=_mm512_sub_ps(t2,_mm512_mul_ps(t1,t3));
                _mm512_store_ps(matrix[i]+j,t4);
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
         avx512OmpGauss(matrix,N[p]);
         }
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"avx512不考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;

    }

}

void setMatrix2(float** matrix)
{
    int N[10]={8,32,128,256,512,1024,2048,3000,4096,5200};
    for(int p=0;p<10;p++)
    {
        matrix =reinterpret_cast<float**>(_mm_malloc(sizeof(float*)*N[p], 64));
        for (int i = 0; i < N[p]; i++)
        {
              matrix[i]  = reinterpret_cast<float*>(_mm_malloc(sizeof(float)*N[p], 64));
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
         avx512OmpGaussAlign(matrix,N[p]);
         }
         gettimeofday(&end,NULL);
         diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
         cout<<"两处均优化+考虑对齐："<<endl;
         cout<<"N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;

    }
}



int main()
{
    float** matrix1;
    setMatrix1(matrix1);
    cout<<endl;
    //float** matrix2;
    //setMatrix2(matrix2);

    return 0;

}

