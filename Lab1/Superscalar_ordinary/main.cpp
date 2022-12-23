#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
using namespace std;

const long int M=pow(2.0,28);
long int a[M];
long int sum;
int main()
{
    srand((unsigned)time(NULL));
    cout<<"Normal Algorithm"<<endl;
    for(int N=2;N<=M;N*=2){
        for(int i=0;i<N;i++){
            a[i]=rand()%100;
        } //initialize
        long long head,tail,freq;
        long counter;
        if(N<=pow(2.0,10))
            counter=1000000;
        else if(N>pow(2.0,10)&&N<=pow(2.0,15))
            counter=100000;
        else if(N>pow(2.0,15)&&N<=pow(2.0,20))
            counter=1000;
        else
            counter=10;
        long c=counter;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        while(c>0){
            c--;
            sum=0;
            for(int i=0;i<N;i++){
                sum+=a[i];
            }/*Normal algorithm*/
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time=(tail-head)*1000/freq;
        cout<<N<<" "<<counter<<" "<<time<<"ms "<<time/counter<<"ms"<<endl;



    }
    return 0;
}

