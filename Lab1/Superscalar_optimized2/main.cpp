#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
using namespace std;

const long int M=pow(2.0,28);
long int a[M];
long int sum;

void recursion(long int a[],int n){
    if(n==1){
        return;
    }
    for(int i=0;i<n/2;i++){
        a[i]+=a[n-i-1];
        n=n/2;
        recursion(a,n);
    }
}
int main()
{
    srand((unsigned)time(NULL));
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
            recursion(a,N);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time=(tail-head)*1000/freq;
        cout<<N<<" "<<counter<<" "<<time<<"ms "<<time/counter<<"ms"<<endl;



    }
    return 0;
}
