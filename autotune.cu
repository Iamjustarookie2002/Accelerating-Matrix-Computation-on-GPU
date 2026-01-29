#include<cuda_runtime.h>
#include<iostream>

const int NUM_THREADS = 256;

template<int BM, int BN, int BK, int TM, int TK>
__global__ void __launch_bounds__(NUM_THREADS)
matrix_mul(int *A,int *B,int *C,int m, int n, int k){
    int cCol=blockIdx.x;
    int cRow=blockIdx.y;

    A+=cRow*n*BM;
    B+=cCol*BK;
    C+=cRow*k*BM+cCol*BK;

    __shared__ int As[BM*BN];
    __shared__ int Bs[BN*BK];
    
    // BELOW IS WARPTILE SIZE , MEANING THE OUTPUT DIMENSION EACH WARP COMPUTES
    // EACH WARP HAS 32 THREADS, 16 IN X DIRECTION AND 16 IN Y DIRECTION 
    constexpr int WM = TM * 16;
    constexpr int WK = TK * 16;
  
    constexpr int WMITER = (BM+WM-1)/WM;
    constexpr int WKITER = (BK+WK-1)/WK;


    int threadRow=threadIdx.x/(WK/TK);
    int threadCol=threadIdx.x%(WK/TK);

    int innerColA = threadIdx.x % (BN/4); 
    int innerRowA = threadIdx.x / (BN/4);
    constexpr int rowStrideA = (NUM_THREADS * 4) / BN;

    int innerColB = threadIdx.x % (BK/4); 
    int innerRowB = threadIdx.x / (BK/4);
    constexpr int rowStrideB = NUM_THREADS / (BK/ 4);
 

    int results[WMITER*WKITER*TM*TK]={0};
    int regA[TM]={0};
    int regB[TK]={0};

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
      
        for(int outer_i=0;outer_i<WMITER;outer_i++){
            for(int outer_j=0;outer_j<WKITER;outer_j++){
                for(int j=0;j<BN;j++){
                for (int idx=0;idx<TM;idx++){
                    regA[idx]=As[j*BM+(outer_i*WM)+threadRow*TM+idx];
                }
                for (int idx=0;idx<TK;idx++){
                    regB[idx]=Bs[j*BK+(outer_j*WK)+threadCol*TK+idx];
                }
                for(int idx=0;idx<TM;idx++){
                    for(int count=0;count<TK;count++){
                    results[(outer_i*TM+idx)*(WMITER*TK)+outer_j*TK+count]+=regA[idx]*regB[count];
                    }
                }
                }
            }
        }
        __syncthreads();
        A+=BN;
        B+=BN*k;
    }

    for(int outer_i=0;outer_i<WMITER;outer_i++){
        for(int outer_j=0;outer_j<WKITER;outer_j++){
            int *C_interim=C+(outer_i*WM*k)+(outer_j*WK);
            for(int i=0;i<TM;i++){
                for(int j=0;j<TK;j+=4){
                int4 tmp = reinterpret_cast<int4 *>(&C_interim[(threadRow*TM+i)*k+threadCol*TK+j])[0];
                const int ref= (outer_i* TM + i) * (WKITER * TK) + outer_j * TK + j;
                tmp.x=results[ref+0];
                tmp.y=results[ref+1];
                tmp.z=results[ref+2];
                tmp.w=results[ref+3];

                reinterpret_cast<int4 *>(&C_interim[(threadRow*TM+i)*k+threadCol*TK+j])[0]=tmp;
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
    const int TM = 4;
    const int TK = 4;
    const int BM = 64;
    const int BK = 64;


    dim3 blockdim(NUM_THREADS);
    dim3 griddim((k+BK-1)/BK,(m+BM-1)/BM);
    matrix_mul<BM, BN, BK, TM, TK><<<griddim,blockdim>>>(dA,dB,dC,m,n,k);

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