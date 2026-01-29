#include<cuda_runtime.h>
#include<iostream>

#define BLOCKSIZE 16
#define THREADCOUNT 4

__global__ void matrix_mul(int *A,int *B,int *C,int m, int n, int k){
    int cCol=blockIdx.x;
    int cRow=blockIdx.y;

    A+=cRow*n*BLOCKSIZE;
    B+=cCol*BLOCKSIZE;
    C+=cRow*k*BLOCKSIZE+cCol*BLOCKSIZE;

    __shared__ int As[BLOCKSIZE*THREADCOUNT];
    __shared__ int Bs[THREADCOUNT*BLOCKSIZE];

    int threadRow=threadIdx.x/BLOCKSIZE;
    int threadCol=threadIdx.x%BLOCKSIZE;

    int innerColA = threadIdx.x % THREADCOUNT; 
    int innerRowA = threadIdx.x / THREADCOUNT;
    int innerColB = threadIdx.x % BLOCKSIZE; 
    int innerRowB = threadIdx.x / BLOCKSIZE;

    int results[THREADCOUNT]={0};
    for(int i=0;i<n;i+=THREADCOUNT){

        As[innerRowA*THREADCOUNT+innerColA]=A[innerRowA*n+innerColA];
        Bs[innerRowB*BLOCKSIZE+innerColB]=B[innerRowB*k+innerColB];

        __syncthreads();

        A+=THREADCOUNT;
        B+=THREADCOUNT*k;

        for(int j=0;j<THREADCOUNT;j++){
            int tempB=Bs[j*BLOCKSIZE+threadCol];
            for(int count=0;count<THREADCOUNT;count++){
            results[count]+=As[(threadRow*THREADCOUNT+count)*THREADCOUNT+j]*tempB;
            }
        }
        __syncthreads();
    }

    for(int i=0;i<THREADCOUNT;i++){
        C[(threadRow*THREADCOUNT+i)*k+threadCol]=results[i];
        }
    }




int main(){
    int m=512;
    int n=512;
    int k=512;
    int *A= new int[m*n];
    int *B= new int[n*k];
    int *C= new int [m*k];

    int *dA,*dB,*dC;

    cudaMalloc((void**)&dA,sizeof(int)*m*n);
    cudaMalloc((void**)&dB,sizeof(int)*n*k);
    cudaMalloc((void**)&dC,sizeof(int)*m*k);
     
    for(int i = 0;i < m;++i){
    for(int j = 0;j < n;++j){
        A[i*n+j] = i+j;
    }
}
    for(int i = 0; i < n; ++i){
    for(int j = 0; j < k; ++j){
        B[i*k+j] = i*j;
    }
}
    cudaMemcpy(dA,A,sizeof(int)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(int)*n*k,cudaMemcpyHostToDevice);
    

    dim3 blockdim((BLOCKSIZE*BLOCKSIZE)/THREADCOUNT);
    dim3 griddim((k+BLOCKSIZE-1)/BLOCKSIZE,(m+BLOCKSIZE-1)/BLOCKSIZE);
    matrix_mul<<<griddim,blockdim>>>(dA,dB,dC,m,n,k);

    cudaMemcpy(C,dC,sizeof(int)*m*k,cudaMemcpyDeviceToHost);

    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
              std::cout << C[i * k + j] << "\t";
        }
        std::cout << std::endl;
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] A;
    delete[] B;
    delete[] C;

}