/*
Game of Life - CUDA implementation

Rules:
1. Any live cell with fewer than 2 live neighbors dies.
2. Any live cell with 2 or 3 live neighbors lives.
3. Any live cell with more than 3 live neighbors dies.
4. Any dead cell with exactly 3 live neighbors becomes a live cell.
*/

#include <iostream>
#include <cuda.h>
#include <unistd.h>
#include <signal.h> 
#include <cstdlib>
#include <chrono>

const char* author = "sencery"; 

// 16x16 grid
const int WIDTH = 96;
const int HEIGHT = 96;
int* h_grid = nullptr;
int* h_newGrid = nullptr;
int* d_grid = nullptr;
int* d_newGrid = nullptr;


__global__ void updateGrid(int* d_grid, int* d_newGrid, int width, int height) {
    // calculating x and y indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // finding the 1D index of the cell
    int idx = y * width + x;

    // check if the cell is within the grid bounds
    if (x < width && y < height) {
        int live_neighbors = 0;

        // check all 8 neighbors of the cell
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue; // this is the cell itself

                int neighbor_x = x + i;
                int neighbor_y = y + j;

                // check if the neighbor is within the grid bounds
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    live_neighbors += d_grid[neighbor_y * width + neighbor_x];
                }
            }
        }

        // set the new state of the cell
        if (d_grid[idx] == 1) { // cell is alive
            if (live_neighbors < 2 || live_neighbors > 3) {
                d_newGrid[idx] = 0; // cell dies
            } else {
                d_newGrid[idx] = 1; // cell lives
            }
        } else { // cell is dead
            if (live_neighbors == 3) {
                d_newGrid[idx] = 1; // cell becomes alive
            } else {
                d_newGrid[idx] = 0; // cell stays dead
            }
        }
    }
}


// cleanup once the program is terminated
void cleanup(int signum) {
    std::cout << "Exiting..." << std::endl;
    
    if (d_grid) cudaFree(d_grid);
    if (d_newGrid) cudaFree(d_newGrid);
    if (h_grid) free(h_grid);
    if (h_newGrid) free(h_newGrid);

    exit(signum);
}

// initialize the grid with random values
// chrono -> https://stackoverflow.com/questions/53040940/why-is-the-new-random-library-better-than-stdrand 
void initializeRandomGrid(int* grid) {
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    srand(seed);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        grid[i] = rand() % 2;
    }
}

// TODO: function to let user initialize the grid manually
void initializeManualGrid(int* grid) {
}


void printGrid(int* grid) {
    std::cout << "\033[H\033[J";  // clearing terminal every time to have only 1 grid 
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (grid[y * WIDTH + x] == 1) {
                std::cout << "O ";
            } else {
                std::cout << ". ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    signal(SIGINT, cleanup);

    int* h_grid = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    int* h_newGrid = (int*)malloc(WIDTH * HEIGHT * sizeof(int));

    initializeRandomGrid(h_grid);

    cudaMalloc((void**)&d_grid, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc((void**)&d_newGrid, WIDTH * HEIGHT * sizeof(int));

    cudaMemcpy(d_grid, h_grid, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    // dim3 -> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=dim3#dim3
    dim3 blockDim(4, 4);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    // main loop
    while (1) { 
        updateGrid<<<gridDim, blockDim>>>(d_grid, d_newGrid, WIDTH, HEIGHT);
        cudaMemcpy(h_newGrid, d_newGrid, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
        printGrid(h_newGrid);
        usleep(300000); 
        cudaMemcpy(d_grid, d_newGrid, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cleanup(0); // never reached but habits die hard
    return 0; 
}