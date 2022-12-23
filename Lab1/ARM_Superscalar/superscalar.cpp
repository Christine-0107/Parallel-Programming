#include<stdio.h>
#include<sys/time.h>
#include<unistd.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>
using namespace std;

const int N=pow(2.0,17);
int a[N];
long int sum;


int ordinary(int a[],int n){
    int ss=0;
    for(int i=0;i<n;i++){
        ss+=a[i];
    }
    return ss;
}

int better1(int a[],int n){
    long int sum1=0,sum2=0;
    for(int i=0;i<n;i+=2){
        sum1+=a[i];
        sum2+=a[i+1];
    }
    return sum1+sum2;
}

void recursion(int a[],int n){
    if(n==1){
        return;
    }
    else{
        for(int i=0;i<n/2;i++){
            a[i]+=a[n-i-1];
        }
        n/=2;
        recursion(a,n);
    }
}

void doubleCycle(int a[],int n){
    for(int m=n;m>1;m/=2){
        for(int i=0;i<m/2;i++){
            a[i]=a[i*2]+a[i*2+1];
        }
    }
}


int main(){
        srand((int)time(0));
        int counter=1;
        for(int i=0;i<N;i++)
            a[i]=rand()%100;

        cout<<"double"<<endl;
            int n=N;
            struct  timeval start;
            struct  timeval end;
            unsigned  long time;
            gettimeofday(&start,NULL);
            doubleCycle(a,n);
            gettimeofday(&end,NULL);
            time = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
            cout<<n<<" "<<counter<<" "<<time<<"us "<<time/counter<<"us"<<endl;


        cout<<endl;
        //better1

//        cout<<"Better1"<<endl;
//        for(int n=2;n<=N;n*=2){
//            struct  timeval start;
//            struct  timeval end;
//            unsigned  long time;
//            if(n<=pow(2.0,10))
//                counter=100000;
//            else if(n>pow(2.0,10)&&n<=pow(2.0,15))
//                counter=10000;
//            else if(n>pow(2.0,15)&&n<=pow(2.0,20))
//                counter=100;
//            else
//                counter=1;
//            int c=counter;
//
//            if(c!=1){
//                gettimeofday(&start,NULL);
//                while(c>0){
//                    c--;
//                    sum=better1(a,n);
//                }
//                gettimeofday(&end,NULL);
//            }
//            else{
//                gettimeofday(&start,NULL);
//                sum=better1(a,n);
//                gettimeofday(&end,NULL);
//            }
//
//            time = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
//
//            cout<<n<<" "<<counter<<" "<<time<<"us "<<time/counter<<"us"<<endl;
//
//        }
//        cout<<endl;
        //better 2_1
//        cout<<"Recursion"<<endl;
//        for(int n=2;n<=N;n*=2){
//            for(int i=0;i<n;i++)
//                a[i]=rand()%100;
//            struct  timeval start;
//            struct  timeval end;
//            unsigned  long time;
//            if(n<=pow(2.0,10))
//                counter=100000;
//            else if(n>pow(2.0,10)&&n<=pow(2.0,15))
//                counter=10000;
//            else if(n>pow(2.0,15)&&n<=pow(2.0,20))
//                counter=100;
//            else
//                counter=1;
//            int c=counter;
//
//            if(c!=1){
//                gettimeofday(&start,NULL);
//                while(c>0){
//                    c--;
//                    recursion(a,n);
//                    sum=a[0];
//                }
//                gettimeofday(&end,NULL);
//            }
//            else{
//                gettimeofday(&start,NULL);
//                recursion(a,n);
//                sum=a[0];
//                gettimeofday(&end,NULL);
//            }
//
//            time = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
//
//            cout<<n<<" "<<counter<<" "<<time<<"us "<<time/counter<<"us"<<endl;
//        }
//        cout<<endl;
        //better 2_2
//        cout<<"Double Cycles"<<endl;
//        for(int n=2;n<=N;n*=2){
//            for(int i=0;i<n;i++)
//                a[i]=rand()%100;
//            struct  timeval start;
//            struct  timeval end;
//            unsigned  long time;
//            if(n<=pow(2.0,10))
//                counter=100000;
//            else if(n>pow(2.0,10)&&n<=pow(2.0,15))
//                counter=10000;
//            else if(n>pow(2.0,15)&&n<=pow(2.0,20))
//                counter=100;
//            else
//                counter=1;
//            int c=counter;
//
//            if(c!=1){
//                gettimeofday(&start,NULL);
//                while(c>0){
//                    c--;
//                    doubleCycle(a,n);
//                    sum=a[0];
//                }
//                gettimeofday(&end,NULL);
//            }
//            else{
//                gettimeofday(&start,NULL);
//                doubleCycle(a,n);
//                sum=a[0];
//                gettimeofday(&end,NULL);
//            }
//
//            time = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
//            cout<<n<<" "<<counter<<" "<<time<<"us "<<time/counter<<"us"<<endl;
//        }
        return 0;
}
