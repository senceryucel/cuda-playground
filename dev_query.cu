#include <stdio.h>
#include <cuda_runtime.h>

int _ConvertSMVer2Cores(int major, int minor, const char **arch_name) {
    typedef struct {
        int SM;
        int Cores;
        const char *name;
    } sSMtoCores;

    // using the SM version to determine the arch (https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h)
    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192, "Kepler"}, // Kepler
        {0x32, 192, "Kepler"},
        {0x35, 192, "Kepler"},
        {0x37, 192, "Kepler"},
        {0x50, 128, "Maxwell"}, // Maxwell
        {0x52, 128, "Maxwell"},
        {0x53, 128, "Maxwell"},
        {0x60, 64,  "Pascal"},  // Pascal
        {0x61, 128, "Pascal"},
        {0x62, 128, "Pascal"},
        {0x70, 64,  "Volta"},   // Volta
        {0x72, 64,  "Volta"},
        {0x75, 64,  "Turing"},  // Turing
        {0x80, 64,  "Ampere"},  // Ampere
        {0x86, 128, "Ampere"},
        {0x87, 128, "Ampere"},
        {-1, -1, "Unknown"}
    };

    int index = 0;

    // find the GPU arch
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            *arch_name = nGpuArchCoresPerSM[index].name;
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    *arch_name = "Unknown";

    return 128; // for future archs
}


int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    printf("Number of CUDA-capable devices: %d\n", deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        const char *arch_name;
        int cudaCoresPerSM = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor, &arch_name);
        int totalCudaCores = cudaCoresPerSM * deviceProp.multiProcessorCount;

        printf("\nDevice %d: \"%s\"\n", device, deviceProp.name);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  Architecture Type:                             %s\n", arch_name);
        printf("  Total amount of global memory:                 %.2f GB\n",
               deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
        printf("  CUDA Cores:                                    %d (Cores per SM: %d)\n", totalCudaCores, cudaCoresPerSM);
        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum grid dimensions:                       [%d, %d, %d]\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Maximum block dimensions:                      [%d, %d, %d]\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }

    return 0;
}