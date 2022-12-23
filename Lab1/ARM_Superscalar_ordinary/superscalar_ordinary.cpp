#include<stdio.h>
#include<sys/time.h>
#include<unistd.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>
using namespace std;

const int N=pow(2.0,28);
int a[N];
long int sum;

int main(){
        srand((int)time(0));
        int counter=1;
        for(int i=0;i<N;i++)
                a[i]=rand()%100;
        for(int n=2;n<=N;n*=2){
            struct  timeval start;
            struct  timeval end;

            unsigned  long time;
            if(n<=pow(2.0,10))
                counter=100000;
            else if(n>pow(2.0,10)&&n<=pow(2.0,15))
                counter=10000;
            else if(n>pow(2.0,15)&&n<=pow(2.0,20))
                counter=100;
            else
                counter=1;
            int c=counter;

            if(c!=1){
                gettimeofday(&start,NULL);
                while(c>0){
                    c--;
                    sum=0;
                    for(int i=0;i<n;i++){
                        sum+=a[i];
                    }
                }
                gettimeofday(&end,NULL);
            }
            else{
                gettimeofday(&start,NULL);
                for(int i=0;i<n;i++){
                        sum+=a[i];
                }
                gettimeofday(&end,NULL);
            }

            time = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
            cout<<n<<" "<<counter<<" "<<time<<"us "<<time/counter<<"us"<<endl;
        }
        return 0;
}

