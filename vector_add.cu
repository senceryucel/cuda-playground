// adds two vectors together
#include <stdio.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024;  // vector size
    size_t size = N * sizeof(float);

    // allocating space for input vectors in host
    float *host_A = (float *)malloc(size);
    float *host_B = (float *)malloc(size);
    float *host_C = (float *)malloc(size);

    // initialize inputs
    for (int i = 0; i < N; i++) {
        host_A[i] = i;
        host_B[i] = i * 3.0f;
    }

    // allocate space for input vectors in device
    float *device_A, *device_B, *device_C;
    cudaMalloc((void **)&device_A, size);
    cudaMalloc((void **)&device_B, size);
    cudaMalloc((void **)&device_C, size);

    // cp inputs from host to device
    cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, N);

    cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (host_C[i] != host_A[i] + host_B[i]) {
            printf("Error: result[%d] = %f\n", i, host_C[i]);
            return -1;
        }
    }

    printf("Test PASSED\n");
    
    // result
    for (int i = 0; i < N; i++) {
        printf("%f ", host_C[i]);
    }

    // free memory
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(host_A);
    free(host_B);
    free(host_C);

    return 0;
}