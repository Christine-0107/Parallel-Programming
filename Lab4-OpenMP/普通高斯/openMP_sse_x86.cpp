#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>
using namespace std;

const int NUM_THREADS=4;

void serialGauss(float** matrix,int N)
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
}

void normalOmpGauss(float** matrix,int N)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for(int k=0;k<N;k++)
    {
	#pragma omp single
	{
		for(int j=k;j<N;j++)
		{
			matrix[k][j]=matrix[k][j]/matrix[k][k];
		}
        }
        #pragma omp for
	for(int i=k+1;i<N;i++)
	{
		for(int j=k+1;j<N;j++)
		{
			matrix[i][j]=matrix[i][j]-matrix[i][k]*matrix[k][j];
		}
		matrix[i][k]=0.0;
	}
    }
}

void sseOmpGauss(float** matrix,int N)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for(int k=0;k<N;k++)
    {
        #pragma omp single
        {
		float temp[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
                __m128 t1=_mm_loadu_ps(temp);
                int j;
		__m128 t2,t3;
		for(j=k;j<=N-4;j+=4)
                {
                        t2=_mm_loadu_ps(matrix[k]+j);
			t3=_mm_div_ps(t2,t1);
			_mm_storeu_ps(matrix[k]+j,t3);
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
        for(int i=k+1;i<N;i++)
        {
		float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
		__m128 t1=_mm_loadu_ps(temp2);
		int j;
                __m128 t2,t3,t4;
                for(j=k+1;j<=N-4;j+=4)
                {
                        t2=_mm_loadu_ps(matrix[i]+j);
			t3=_mm_loadu_ps(matrix[k]+j);
			t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
			_mm_storeu_ps(matrix[i]+j,t4);
                }
		if(j<N)
		{
			for(;j<N;j++)
			{
				matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
			}
		}
                matrix[i][k]=0.0;
        }
    }
}

void sseOmpGaussAlign(float** matrix,int N)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for(int k=0;k<N;k++)
    {
        #pragma omp single
        {
		float temp[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
                __m128 t1=_mm_load_ps(temp);
                if(k%4!=0){ //先处理前面不对齐的元素
            		for(int p=k;p<k+4-k%4;p++){
                		matrix[k][p]=matrix[k][p]/temp[0];
            		}
        	}

		int j;
		__m128 t2,t3;
		for(j=k+4-k%4;j<=N-4;j+=4)
                {
                        t2=_mm_load_ps(matrix[k]+j);
			t3=_mm_div_ps(t2,t1);
			_mm_store_ps(matrix[k]+j,t3);
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
        for(int i=k+1;i<N;i++)
        {
		float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
		__m128 t1=_mm_load_ps(temp2);
		if((k+1)%4!=0){
             		for(int p=k+1;p<(k+1)+4-(k+1)%4;p++){
                	    matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
                	}
            	}
		int j;
                __m128 t2,t3,t4;
                for(j=k+1+4-(k+1)%4;j<=N-4;j+=4)
                {
                        t2=_mm_load_ps(matrix[i]+j);
			t3=_mm_load_ps(matrix[k]+j);
			t4=_mm_sub_ps(t2,_mm_mul_ps(t1,t3));
			_mm_store_ps(matrix[i]+j,t4);
                }
		if(j<N)
		{
			for(;j<N;j++)
			{
				matrix[i][j]=matrix[i][j]-temp2[0]*matrix[k][j];
			}
		}
                matrix[i][k]=0.0;
        }
    }
}

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
    int N[10]={8,32,128,256,512,1024,2048,3000,4096,5200};
    for(int p=0;p<10;p++)
    {
    	float** matrix = new float* [N[p]];
        for (int i = 0; i < N[p]; i++)
        {
            matrix[i] = new float[N[p]];
        }
        setMatrix(matrix,N[p]);
	struct timeval start;
	struct timeval end;
	unsigned long diff;

	gettimeofday(&start,NULL);
	for(int count=0;count<3;count++)
	{
	serialGauss(matrix,N[p]);
	}
	gettimeofday(&end,NULL);
	diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
	cout<<"串行"<<endl;
	cout<<"N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;

	setMatrix(matrix,N[p]);
	gettimeofday(&start,NULL);
	for(int count=0;count<3;count++)
	{
	normalOmpGauss(matrix,N[p]);
	}
	gettimeofday(&end,NULL);
	diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
	cout<<"普通omp"<<endl;
	cout<<"NUM_THREADS: "<<NUM_THREADS<<" N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;	

	setMatrix(matrix,N[p]);
	gettimeofday(&start,NULL);
	for(int count=0;count<3;count++)
	{
	sseOmpGauss(matrix,N[p]);
	}
	gettimeofday(&end,NULL);
	diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
	cout<<"添加sse不对齐"<<endl;
	cout<<"NUM_THREADS: "<<NUM_THREADS<<" N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;

	setMatrix(matrix,N[p]);
        gettimeofday(&start,NULL);
        for(int count=0;count<3;count++)
        {
        sseOmpGaussAlign(matrix,N[p]);
        }
        gettimeofday(&end,NULL);
        diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
        cout<<"添加sse不对齐"<<endl;
        cout<<"NUM_THREADS: "<<NUM_THREADS<<" N: "<<N[p]<<" time: "<<diff/3<<"us"<<endl;

	cout<<endl;
    }
    return 0;
}
