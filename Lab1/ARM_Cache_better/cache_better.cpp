#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
using namespace std;

const int N=10000;
int a[N];
int b[N][N];
int sum[N];

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
                    for(int i=0;i<n;i++){
                        sum[i]=0;
                    }
                    for(int j=0;j<n;j++){
                        for(int i=0;i<n;i++)
                            sum[i]+=b[j][i]*a[j];
                    }
                }
                gettimeofday(&end,NULL);
            }
            else{
                gettimeofday(&start,NULL);
                for(int i=0;i<n;i++){
                    sum[i]=0;
                }
                for(int j=0;j<n;j++){
                    for(int i=0;i<n;i++)
                        sum[i]+=b[j][i]*a[j];
                }
                gettimeofday(&end,NULL);
            }

            time = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
            cout<<n<<" "<<counter<<" "<<time<<"us "<<time/counter<<"us"<<endl;
            if(n==100)
                step=100;
            if(n==1000)
                step=1000;
        }
        return 0;
}
