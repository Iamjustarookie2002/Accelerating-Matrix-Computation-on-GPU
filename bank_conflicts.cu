#include<cuda_runtime.h>
#include<iostream>

#define BLOCKSIZE 256
#define THREADCOUNT 8

__global__ void matrix_mul(int *A,int *B,int *C,int m, int n, int k){
    int cCol=blockIdx.x;
    int cRow=blockIdx.y;

    A+=cRow*n*BLOCKSIZE;
    B+=cCol*BLOCKSIZE;
    C+=cRow*k*BLOCKSIZE+cCol*BLOCKSIZE;

    __shared__ int As[BLOCKSIZE*THREADCOUNT];
    __shared__ int Bs[THREADCOUNT*BLOCKSIZE];
    

    int threadRow=threadIdx.x/(BLOCKSIZE/THREADCOUNT);
    int threadCol=threadIdx.x%(BLOCKSIZE/THREADCOUNT);

    int innerColA = threadIdx.x % (THREADCOUNT/4); 
    int innerRowA = threadIdx.x / (THREADCOUNT/4);
    int innerColB = threadIdx.x % (BLOCKSIZE/4); 
    int innerRowB = threadIdx.x / (BLOCKSIZE/4);
 

    int results[THREADCOUNT*THREADCOUNT]={0};
    int regA[THREADCOUNT]={0};
    int regB[THREADCOUNT]={0};

    for(int i=0;i<n;i+=THREADCOUNT){
      int4 tmp = reinterpret_cast<int4 *>(&A[innerRowA * n + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * BLOCKSIZE + innerRowA] = tmp.x;
      As[(innerColA * 4 + 1) * BLOCKSIZE + innerRowA] = tmp.y;
      As[(innerColA * 4 + 2) * BLOCKSIZE + innerRowA] = tmp.z;
      As[(innerColA * 4 + 3) * BLOCKSIZE + innerRowA] = tmp.w;
    
      tmp = reinterpret_cast<int4 *>(&B[innerRowB * k + innerColB * 4])[0];
      Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
      Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
      Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
      Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
        __syncthreads();

        A+=THREADCOUNT;
        B+=THREADCOUNT*k;

        for(int j=0;j<THREADCOUNT;j++){
           for (int idx=0;idx<THREADCOUNT;idx++){
            regA[idx]=As[j*BLOCKSIZE+threadRow*THREADCOUNT+idx];
           }
           for (int idx=0;idx<THREADCOUNT;idx++){
            regB[idx]=Bs[(j*8+idx)*16+threadCol];
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
        for(int j=0;j<THREADCOUNT;j+=4){
        int4 tmp = reinterpret_cast<int4 *>(&C[(threadRow*THREADCOUNT+i)*k+threadCol*THREADCOUNT+j])[0];
        tmp.x=results[i*THREADCOUNT+j];
        tmp.y=results[i*THREADCOUNT+j+1];
        tmp.z=results[i*THREADCOUNT+j+2];
        tmp.w=results[i*THREADCOUNT+j+3];

        reinterpret_cast<int4 *>(&C[(threadRow*THREADCOUNT+i)*k+threadCol*THREADCOUNT+j])[0]=tmp;
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