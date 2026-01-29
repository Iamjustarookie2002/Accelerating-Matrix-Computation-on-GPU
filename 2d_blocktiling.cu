#include<cuda_runtime.h>
#include<iostream>

#define BLOCKSIZE 128
#define THREADCOUNT 8

__global__ void matrix_mul(int *A,int *B,int *C,int m, int n, int k){
    int cCol=blockIdx.x;
    int cRow=blockIdx.y;

    A+=cRow*n*BLOCKSIZE;
    B+=cCol*BLOCKSIZE;
    C+=cRow*k*BLOCKSIZE+cCol*BLOCKSIZE;

    __shared__ int As[BLOCKSIZE*THREADCOUNT];
    __shared__ int Bs[THREADCOUNT*BLOCKSIZE];
    
    // each thread computes 8*8 so our indexing should be such that it covers all rows,cols.
    int threadRow=threadIdx.x/(BLOCKSIZE/THREADCOUNT);
    int threadCol=threadIdx.x%(BLOCKSIZE/THREADCOUNT);

    int innerColA = threadIdx.x % THREADCOUNT; 
    int innerRowA = threadIdx.x / THREADCOUNT;
    int innerColB = threadIdx.x % BLOCKSIZE; 
    int innerRowB = threadIdx.x / BLOCKSIZE;

    // There are 256 threads to fill 1024 elements, so each thread must load 4 elements 
    // here we stride across rows, so strideoffset is given by dividing numof threads with number of columns 
    // thread 0 access-[0,0],[32,0],[64,0],[96,0]
    int numberofthreads=(BLOCKSIZE*BLOCKSIZE)/(THREADCOUNT*THREADCOUNT);
    int strideA=numberofthreads/THREADCOUNT;
    int strideB= numberofthreads/BLOCKSIZE;

    int results[THREADCOUNT*THREADCOUNT]={0};
    int regA[THREADCOUNT]={0};
    int regB[THREADCOUNT]={0};

    for(int i=0;i<n;i+=THREADCOUNT){
        for(int j=0;j<BLOCKSIZE;j+=strideA){
            As[(innerRowA+j)*THREADCOUNT+innerColA]=A[(innerRowA+j)*n+innerColA];
        }
        for(int j=0;j<THREADCOUNT;j+=strideB){
            Bs[(innerRowB+j)*BLOCKSIZE+innerColB]=B[(innerRowB+j)*k+innerColB];
        }

        __syncthreads();

        A+=THREADCOUNT;
        B+=THREADCOUNT*k;

        for(int j=0;j<THREADCOUNT;j++){
           for (int idx=0;idx<THREADCOUNT;idx++){
            regA[idx]=As[(threadRow*THREADCOUNT+idx)*THREADCOUNT+j];
           }
           for (int idx=0;idx<THREADCOUNT;idx++){
            regB[idx]=Bs[j*BLOCKSIZE+threadCol*THREADCOUNT+idx];
           }
           for(int idx=0;idx<THREADCOUNT;idx++){
            for(int count=0;count<THREADCOUNT;count++){
            results[idx*THREADCOUNT+count]+=regA[idx]*regB[count];
            }
           }
        }
        __syncthreads();
    }

    for(int i=0;i<THREADCOUNT;i++){
        for(int j=0;j<THREADCOUNT;j++){
        C[(threadRow*THREADCOUNT+i)*k+threadCol*THREADCOUNT+j]=results[i*THREADCOUNT+j];
        }
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
    

    dim3 blockdim((BLOCKSIZE*BLOCKSIZE)/(THREADCOUNT*THREADCOUNT));
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