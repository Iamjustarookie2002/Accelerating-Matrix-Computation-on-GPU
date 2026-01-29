#include<cuda_runtime.h>
#include<iostream>

__global__ void matrix_mul(int *A,int *B,int *C,int m, int n, int k){
    int row=blockIdx.x*blockDim.x+ threadIdx.x;
    int col=blockDim.y*blockIdx.y+threadIdx.y;
    if (row<m  && col<k){
        int sum=0;
        for(int i=0;i<n;i++){
            sum+=A[row*n+i]*B[i*k+col];
        }
        C[row*k+col]=sum;
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
     
    for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        A[i * n + j] = i + j;
    }
}
    for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
        B[i * k + j] = i * j;
    }
}
    cudaMemcpy(dA,A,sizeof(int)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,sizeof(int)*n*k,cudaMemcpyHostToDevice);
    
    int blocksize=16;
    dim3 blockdim(blocksize,blocksize);
    dim3 griddim((m+blocksize-1)/blocksize,(k+blocksize-1)/blocksize);
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