#include<cuda_runtime.h>
#include<iostream>

const int WARPSIZE=32;

template<int BM, int BN, int BK, int WM, int WK, int WKITER,int TM, int TK, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matrix_mul(int *A,int *B,int *C,int m, int n, int k){
    int cCol=blockIdx.x;
    int cRow=blockIdx.y;

    const int warpIdx = threadIdx.x / WARPSIZE; 
    const int warpCol = warpIdx % (BK / WK);
    const int warpRow = warpIdx / (BK / WK);

    
    constexpr int WMITER = (WM * WK) / (WARPSIZE * TM * TK * WKITER);
    constexpr int WSUBM = WM / WMITER; 
    constexpr int WSUBK = WK / WKITER; 

   
    const int threadIdxInWarp = threadIdx.x % WARPSIZE;         
    const int threadColInWarp = threadIdxInWarp % (WSUBK / TK); 
    const int threadRowInWarp = threadIdxInWarp / (WSUBK / TK); 


    A+=cRow*n*BM;
    B+=cCol*BK;
    C += (cRow * BM + warpRow * WM) * k + cCol * BK + warpCol * WK;


    __shared__ int As[BM*BN];
    __shared__ int Bs[BN*BK];
    
    // BELOW IS WARPTILE SIZE , MEANING THE OUTPUT DIMENSION EACH WARP COMPUTES
    // EACH WARP HAS 32 THREADS, 16 IN X DIRECTION AND 16 IN Y DIRECTION 


    int innerColA = threadIdx.x % (BN/4); 
    int innerRowA = threadIdx.x / (BN/4);
    constexpr int rowStrideA = (NUM_THREADS * 4) / BN;

    int innerColB = threadIdx.x % (BK/4); 
    int innerRowB = threadIdx.x / (BK/4);
    constexpr int rowStrideB = NUM_THREADS / (BK/ 4);
 

    int results[WMITER*WKITER*TM*TK]={0};
    int regA[WMITER*TM]={0};
    int regB[WKITER*TK]={0};

    for(int i=0;i<n;i+=BN){
        for(int j=0; j+rowStrideA<=BM;j+=rowStrideA){
      int4 tmp = reinterpret_cast<int4 *>(&A[(innerRowA+j)*n+innerColA*4])[0];
      As[(innerColA * 4 + 0) * BM + innerRowA+j] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA+j] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA+j] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA+j] = tmp.w;
        }

        for(int j=0;j+rowStrideB<=BN;j+=rowStrideB){
      reinterpret_cast<int4 *>(&Bs[(innerRowB+j)*BK+innerColB*4])[0]= reinterpret_cast<int4 *>(&B[(innerRowB+j)*k+innerColB*4])[0];
      
        }
        __syncthreads();
            for (int dotIdx = 0; dotIdx < BN; ++dotIdx) {
            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (int i = 0; i < TM; ++i) {
                regA[wSubRowIdx * TM + i] =
                    As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                    threadRowInWarp * TM + i];
            }
            }
            for (int wSubColIdx = 0; wSubColIdx < WKITER; ++wSubColIdx) {
            for (int i = 0; i < TK; ++i) {
                regB[wSubColIdx * TK + i] =
                    Bs[(dotIdx * BK) + warpCol * WK + wSubColIdx * WSUBK +
                    threadColInWarp * TK + i];
            }
            }

            for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (int wSubColIdx = 0; wSubColIdx < WKITER; ++wSubColIdx) {
                for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TK; ++resIdxN) {
                    results[(wSubRowIdx * TM + resIdxM) * (WKITER * TK) +
                                (wSubColIdx * TK) + resIdxN] +=
                        regA[wSubRowIdx * TM + resIdxM] *
                        regB[wSubColIdx * TK + resIdxN];
                }
                }
            }
            }
        }
     
        A+=BN;
        B+=BN*k;
         __syncthreads();
    }

    for(int outer_i=0;outer_i<WMITER;outer_i++){
        for(int outer_j=0;outer_j<WKITER;outer_j++){
            int *C_interim=C+(outer_i*WSUBM)*k+(outer_j*WSUBK);
            for(int i=0;i<TM;i++){
                for(int j=0;j<TK;j+=4){
                int4 tmp = reinterpret_cast<int4 *>(&C_interim[(threadRowInWarp*TM+i)*k+threadColInWarp*TK+j])[0];
                const int ref= (outer_i* TM + i) * (WKITER * TK) + outer_j * TK + j;
                tmp.x=results[ref+0];
                tmp.y=results[ref+1];
                tmp.z=results[ref+2];
                tmp.w=results[ref+3];

                reinterpret_cast<int4 *>(&C_interim[(threadRowInWarp*TM+i)*k+threadColInWarp*TK+j])[0]=tmp;
                }
            }
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
    
    //best params
    const int BN = 32;
    const int TM = 8;
    const int TK = 4;
    const int BM = 128;
    const int BK = 128;
    const int WM=32;
    const int WK=128;
    const int NUM_THREADS=128;
    const int WKITER=4;



    dim3 blockdim(NUM_THREADS);
    dim3 griddim((k+BK-1)/BK,(m+BM-1)/BM);
    matrix_mul<BM, BN, BK, WM,WK,WKITER,TM, TK,NUM_THREADS><<<griddim,blockdim>>>(dA,dB,dC,m,n,k);

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