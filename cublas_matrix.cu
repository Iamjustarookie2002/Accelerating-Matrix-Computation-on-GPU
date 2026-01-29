#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

int main() {
    const int m = 512;
    const int n = 512;
    const int k = 512;

    float *A = new float[m * n];
    float *B = new float[n * k];
    float *C = new float[m * k];

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[i * n + j] = static_cast<float>(i + j);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            B[i * k + j] = static_cast<float>(i * j);

    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, sizeof(float) * m * n);
    cudaMalloc((void**)&dB, sizeof(float) * n * k);
    cudaMalloc((void**)&dC, sizeof(float) * m * k);

    cudaMemcpy(dA, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * n * k, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        k,
        m,
        n,
        &alpha,
        dB, k,
        dA, n,
        &beta,
        dC, k
    );

    cudaMemcpy(C, dC, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
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
    cublasDestroy(handle);

    return 0;
}
