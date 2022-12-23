#include <arm_neon.h>
#include <iostream>
#include <sys/time.h>
using namespace std;

//两处均优化
//对齐
float** parallelGaussAlign(float** matrix,int N)
{
	float32x4_t t1,t2,t3,t4;
	for(int k=0;k<N;k++)
	{
		float digonal[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
		t1=vld1q_f32(digonal);
		if((k+k*N)%4!=0)
		{
			for(int p=k;p<k+4-(k+k*N)%4;p++)
			{
				matrix[k][p]=matrix[k][p]/digonal[0];
			}
		}
		int j;
		for(j=k+4-(k+k*N)%4;j<=N-4;j+=4){
			t2=vld1q_f32(matrix[k]+j);
			t3=vdivq_f32(t2,t1);
			vst1q_f32(matrix[k]+j,t3);
		}
		if(j<N)
		{
			for(;j<N;j++)
			{
				matrix[k][j]=matrix[k][j]/digonal[0];
			}
		}
		for(int i=k+1;i<N;i++)
		{
			float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
			t1=vld1q_f32(temp2);
			if((k+1+(k+1)*N)%4!=0){
				for(int p=k+1;p<(k+1)+4-(k+1+(k+1)*N)%4;p++){
					matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
				}
			}
			for(j=(k+1)+4-(k+1+(k+1)*N)%4;j<=N-4;j+=4){
				t2=vld1q_f32(matrix[i]+j);
				t3=vld1q_f32(matrix[k]+j);
				t4=vsubq_f32(t2,vmulq_f32(t1,t3));
				vst1q_f32(matrix[i]+j,t4);
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
//考虑对齐
float** firstParallelAlign(float** matrix,int N)
{
	float32x4_t t1,t2,t3,t4;
	for(int k=0;k<N;k++){
		float diagonal[4]={matrix[k][k],matrix[k][k],matrix[k][k],matrix[k][k]};
		t1=vld1q_f32(diagonal);
		if((k+k*N)%4!=0)
		{
			for(int p=k;p<k+4-(k+k*N)%4;p++){
				matrix[k][p]=matrix[k][p]/diagonal[0];
			}
		}
		int j;
		for(j=k+4-(k+k*N)%4;j<=N-4;j+=4){
			t2=vld1q_f32(matrix[k]+j);
			t3=vdivq_f32(t2,t1);
			vst1q_f32(matrix[k]+j,t3);
		}
		if(j<N){
			for(;j<N;j++){
				matrix[k][j]=matrix[k][j]/diagonal[0];
			}
		}
		for(int i=k+1;i<N;i++){
			float temp=matrix[i][k];
			for(int j=k+1;j<N;j++){
				matrix[i][j]=matrix[i][j]-temp*matrix[k][j];
			}
			matrix[i][k]=0;
		}
	}
	return matrix;
}

//只优化第二处
//对齐
float** secondParallelAlign(float** matrix,int N){
	float32x4_t t1,t2,t3,t4;
	for(int k=0;k<N;k++)
	{
		float pivot=matrix[k][k];
		for(int j=k;j<N;j++)
		{
			matrix[k][j]=matrix[k][j]/pivot;
		}
		for(int i=k+1;i<N;i++){
			float temp2[4]={matrix[i][k],matrix[i][k],matrix[i][k],matrix[i][k]};
			t1=vld1q_f32(temp2);
			if((k+1+(k+1)*N)%4!=0){
				for(int p=k+1;p<(k+1)+4-(k+1+(k+1)*N)%4;p++){
					matrix[i][p]=matrix[i][p]-temp2[0]*matrix[k][p];
				}
			}
			int j;
			for(j=(k+1)+4-(k+1+(k+1)*N)%4;j<=N-4;j+=4){
				t2=vld1q_f32(matrix[i]+j);
				t3=vld1q_f32(matrix[k]+j);
				t4=vsubq_f32(t2,vmulq_f32(t1,t3));
				vst1q_f32(matrix[i]+j,t4);
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

void setMatrix(float** matrix,int N){
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
	int N[8]={8,32,128,256,512,1024,2048,4096};
	for(int p=0;p<8;p++){
		float** matrix=new float*[N[p]];
		for(int i=0;i<N[p];i++){
			matrix[i]=new float[N[p]];
		}
	
	setMatrix(matrix,N[p]);
	struct timeval start;
	struct timeval end;
	unsigned long diff;
	gettimeofday(&start,NULL);
	float** M2=parallelGaussAlign(matrix,N[p]);
	gettimeofday(&end,NULL);
	diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
	cout<<"两处均优化+考虑对齐："<<endl;
	cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;

	setMatrix(matrix,N[p]);
	gettimeofday(&start,NULL);
	float** M4=firstParallelAlign(matrix,N[p]);
	gettimeofday(&end,NULL);
	diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
	cout<<"优化一处+考虑对齐："<<endl;
	cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;

	setMatrix(matrix,N[p]);
	gettimeofday(&start,NULL);
	float** M6=secondParallelAlign(matrix,N[p]);
	gettimeofday(&end,NULL);
	diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
	cout<<"优化二处+考虑对齐："<<endl;
	cout<<"N: "<<N[p]<<" time: "<<diff<<"us"<<endl;
	cout<<endl;
	}
	return 0;
}
