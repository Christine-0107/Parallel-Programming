#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <cmath>
using namespace std;

const long int M=pow(2.0,28);
long int a[M];
long int sum;
int main()
{

    cout<<"Normal Algorithm"<<endl;
    //int step=10;
    for(int N=2;N<=M;N*=2){
        for(int i=0;i<N;i++){
            a[i]=rand()%100;
        } //initialize
        long long head,tail,freq;
        long counter;
        if(N<=pow(2.0,10))
            counter=100000;
        else if(N>pow(2.0,10)&&N<=pow(2.0,15))
            counter=10000;
        else if(N>pow(2.0,15)&&N<=pow(2.0,20))
            counter=100;
        else
            counter=1;
        long c=counter;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        while(c>0){
            c--;
            for(int k=N;k>1;k/=2){
                for(int i=0;i<k/2;i++){
                    a[i]=a[i*2]+a[i*2+1];
                }
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time=(tail-head)*1000/freq;
        cout<<N<<" "<<counter<<" "<<time<<"ms "<<time/counter<<"ms"<<endl;



    }
    return 0;
}
