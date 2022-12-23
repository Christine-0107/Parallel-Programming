#include<stdio.h>
#include<sys/time.h>
#include<unistd.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

const int N=10000;
int a[N];
int b[N][N];
int sum_1[N];
int sum_2[N];

void ordinary(int a[],int b[][N],int sum_1[],int n){
    for(int i=0;i<n;i++){
        sum_1[i]=0;
        for(int j=0;j<n;j++){
            sum_1[i]+=b[j][i]*a[j];
        }
    }
}

void better(int a[],int b[][N],int sum_2[],int n){
    for(int i=0;i<n;i++){
        sum_2[i]=0;
    }
    for(int j=0;j<n;j++){
        for(int i=0;i<n;i++){
            sum_2[i]+=b[j][i]*a[j];
        }
    }
}



int main(){
        srand((int)time(0));
        int counter=1;
        for(int i=0;i<N;i++)
                a[i]=rand()%100;
        for(int i=0;i<N;i++)
                for(int j=0;j<N;j++)
                        b[i][j]=rand()%100;
        int step=10;
        for(int n=0;n<=10000;n+=step){
            struct  timeval start;
            struct  timeval end;
            unsigned  long time;
            if(n<=50)
                counter=100000;
            else if(n>50&&n<=100)
                counter=10000;
            else if(n>100&&n<=1000)
                counter=100;
            else
                counter=1;
            int c=counter;

            if(c!=1){
                gettimeofday(&start,NULL);
                while(c>0){
                    c--;
                    ordinary(a,b,sum_1,n);
                }
                gettimeofday(&end,NULL);
            }
            else{
                gettimeofday(&start,NULL);
                ordinary(a,b,sum_1,n);
                gettimeofday(&end,NULL);
            }
            time = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
            cout<<n<<" "<<counter<<" "<<time<<"us "<<time/counter<<"us"<<endl;
            if(n==100)
                step=100;
            if(n==1000)
                step=1000;
        }
        int step1=10;
        for(int n=0;n<=10000;n+=step1){
            struct  timeval start1;
            struct  timeval end1;
            unsigned  long time1;
            if(n<=50)
                counter=100000;
            else if(n>50&&n<=100)
                counter=10000;
            else if(n>100&&n<=1000)
                counter=100;
            else
                counter=1;
            int c1=counter;

            if(c1!=1){
                gettimeofday(&start1,NULL);
                while(c1>0){
                    c1--;
                    better(a,b,sum_2,n);
                }
                gettimeofday(&end1,NULL);
            }
            else{
                gettimeofday(&start1,NULL);
                better(a,b,sum_2,n);
                gettimeofday(&end1,NULL);
            }
            time1 = 1000000 * (end1.tv_sec-start1.tv_sec)+ end1.tv_usec-start1.tv_usec;
            cout<<n<<" "<<counter<<" "<<time1<<"us "<<time1/counter<<"us"<<endl;
            if(n==100)
                step1=100;
            if(n==1000)
                step1=1000;
        }
        return 0;
}
