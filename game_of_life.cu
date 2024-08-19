#include <stdio.h>
#include <cuda_runtime.h>

#define N 32  // grid

__global__ void gameOfLife(int *input, int *output) {
}

int main() {
    int *host_input = (int *)malloc(N * N * sizeof(int));
    int *host_output = (int *)malloc(N * N * sizeof(int));

    return 0;
}
