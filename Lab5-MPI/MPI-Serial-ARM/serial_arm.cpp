#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <mpi.h>
#include <arm_neon.h>
#include <cmath>
using namespace std;

void setMatrix(float** matrix,int n)
{
    srand((unsigned)time(0));
    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++){
            matrix[i][j]=0.0;
        }
        matrix[i][i]=1.0;
        for(int j=i+1;j<n;j++){
            matrix[i][j]=rand()%100;
        }
    }
}

void serialGauss(float** matrix,int n)
{
    setMatrix(matrix,n);
    struct timeval start;
    struct timeval end;
    unsigned long diff;
    gettimeofday(&start, NULL);
    for (int k = 0; k < n; k++)
    {
        float  pivot = matrix[k][k];
        for (int j = k; j < n; j++)
        {
            matrix[k][j] = matrix[k][j] / pivot;
        }
        for (int i = k + 1; i < n; i++)
        {
            float temp = matrix[i][k];
            for (int j = k + 1; j < n; j++)
            {
                matrix[i][j] = matrix[i][j] - temp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    gettimeofday(&end, NULL);
    diff=1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec;
    cout<<"serial"<<endl;
    cout<<"N: "<<n<<" time: "<<diff<<"us"<<endl;
}

int main()
{
    int N[10]={8,32,128,256,512,1024,2048,3000,4096,5200};
    MPI_Init(NULL,NULL);
    for(int p=0;p<10;p++)
    {

        float** matrix = new float* [N[p]];
        for (int i = 0; i < N[p]; i++)
        {
            matrix[i] = new float[N[p]];
        }
        serialGauss(matrix,N[p]);


    }
    MPI_Finalize();
    return 0;

}

