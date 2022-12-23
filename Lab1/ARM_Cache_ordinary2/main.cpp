#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
using namespace std;

int a[10000];
int b[10000][10000];
int sum[10000];
int main()
{
    srand((unsigned)time(NULL));
    cout<<"Normal Algorithm"<<endl;
    int step=10;
    int N=10000;
    //for(int N=0;N<=10000;N+=step){
        for(int i=0;i<N;i++){
            a[i]=i;
            for(int j=0;j<N;j++){
                b[i][j]=i+j;
            }
        } //initialize
        struct timeval starts;
        struct timeval ends;

        long counter=1;
        /*if(N<=30)
            counter=10000;
        else if(N>30&&N<=100)
            counter=1000;
        else if(N>100&&N<=1000)
            counter=100;
        else if(N>1000&&N<=5000)
            counter=10;
        else
            counter=1;*/
        long c=counter;
        gettimeofday(&starts,NULL);
        //while(c>0){
            //c--;
            for(int i=0;i<N;i++){
                sum[i]=0;
                for(int j=0;j<N;j++){
                    sum[i]+=b[j][i]*a[j];
                }
            }/*Normal algorithm*/
        //}
        gettimeofday(&ends,NULL);
        unsigned long diff=(ends.tv_sec-starts.tv_sec)*100000+ends.tv_usec-starts.tv_usec;

        cout<<N<<" "<<counter<<" "<<diff<<"us "<<diff/counter<<"us"<<endl;

        /*if(N==100){
            step=100;
        }
        if(N==1000){
            step=1000;
        }*/

    //}
    return 0;
}
