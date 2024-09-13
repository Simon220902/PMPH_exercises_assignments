#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 300

// MAPPING FUNCTION \ x → (x/(x-2.3))^3
__global__ void mapKernel(float* X, float *Y, int N) {
    const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    if (gid < N) {
        float x = X[gid];
        float partial_res = (x/(x-2.3));
        Y[gid] = partial_res*partial_res*partial_res;
    }
}

void mapCPU(float* X, float *Y, int N) {
    for (int i = 0; i < N; i++) {
        float x = X[i];
        Y[i] = (x/(x-2.3f)) * (x/(x-2.3f)) * (x/(x-2.3f));
    }
}

int main(int argc, char** argv) {
    unsigned int N;
    
    { // reading the number of elements 
      if (argc != 2) { 
        printf("Num Args is: %d instead of 1. Exiting!\n", argc); 
        exit(1);
      }

      N = atoi(argv[1]);
      printf("N is: %d\n", N);

      const unsigned int maxN = 1073083647; //replaced 500000000; with 1073083647 since that was the largest input which did not give a core dump.
      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }

    // use the first CUDA device:
    cudaSetDevice(0);

    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = 1.0+(float)i;
    }

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    printf("RUNNING GPU:\n");

    unsigned int B = 256; // chose a suitable block size in dimension x
    unsigned int numblocks = (N + B -1) / B; // number of blocks in dimension x
    dim3 block(B,1,1), grid(numblocks,1,1); //totalnumberofthreads(numblocks*B)mayovershootN!

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        mapKernel<<<grid , block>>>(d_in , d_out, N);
    }

    double gpu_elapsed; struct timeval gpu_t_start, gpu_t_end, gpu_t_diff;
    gettimeofday(&gpu_t_start, NULL);

    for(int r = 0; r < GPU_RUNS; r++) {
        mapKernel<<<grid , block>>>(d_in , d_out, N);
    }
    cudaDeviceSynchronize();

    // TIMING   
    gettimeofday(&gpu_t_end, NULL);
    timeval_subtract(&gpu_t_diff, &gpu_t_end, &gpu_t_start);
    gpu_elapsed = (1.0 * (gpu_t_diff.tv_sec*1e6+gpu_t_diff.tv_usec)) / GPU_RUNS;
    double gpu_gigabytespersec = (2.0 * N * 4.0) / (gpu_elapsed * 1000.0);
    printf("The kernel took on average %f microseconds. GB/sec: %f \n", gpu_elapsed, gpu_gigabytespersec);
    
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    printf("RUNNING CPU:\n");
    // Setup
    float* cpu_X  = (float*) malloc(mem_size);
    float* cpu_Y = (float*) malloc(mem_size);
    for(unsigned int i=0; i<N; ++i) {
        cpu_X[i] = 1.0+(float)i;
    }
    // Timing
    double cpu_elapsed; struct timeval cpu_t_start, cpu_t_end, cpu_t_diff;
    gettimeofday(&cpu_t_start, NULL);
    mapCPU(cpu_X, cpu_Y, N);
    gettimeofday(&cpu_t_end, NULL);
    timeval_subtract(&cpu_t_diff, &cpu_t_end, &cpu_t_start);
    cpu_elapsed = (1.0 * (cpu_t_diff.tv_sec*1e6+cpu_t_diff.tv_usec));
    double cpu_gigabytespersec = (2.0 * N * 4.0) / (cpu_elapsed * 1000.0);
    printf("The CPU took on average %f microseconds. GB/sec: %f \n", cpu_elapsed, cpu_gigabytespersec);

    printf("The acceleration (speedup) from CPU to GPU was: %f\n", (cpu_elapsed/gpu_elapsed));
    int validation_succesful = 1;
    for(unsigned int i=0; i<N; ++i) {
        float actual = h_out[i];
        float expected = cpu_Y[i]; //\ x → (x/(x-2.3))^3

        // (modulo an epsilon error, e.g., fabs(cpu_res[i] - gpu-res[i]) < 0.0001 for all i),
        // and print a VALID or INVALID message. Also make your program print:
        // the runtimes of your CUDA vs CPU-sequential implementation
        // (for CUDA please exclude the time for CPU-to-GPU transfer and GPU memory allocation),
        float epsilon_error = 0.000001;
        if( fabs(expected - actual) > epsilon_error) { // (modulo an epsilon error, e.g., fabs(cpu_res[i] - gpu-res[i]) < 0.0001 for all i),
            printf("Invalid result at index %d, actual: %f, expected: %f. \n", i, actual, expected);
            validation_succesful = 0;
        }
    }
    if (validation_succesful) {
        printf("Successful Validation.\n");
    }

    // clean-up memory
    free(cpu_X);      free(cpu_Y);
    free(h_in);       free(h_out);
    cudaFree(d_in);   cudaFree(d_out);
}
