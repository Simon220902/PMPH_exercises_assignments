// Task 3: CUDA exercise, see lab 1 slides: Lab1-CudaIntro (3pts)
// Write a CUDA program with two functions that both map the function \ x → (x/(x-2.3))^3
// to the array [1,…​,753411], i.e., of size 753411.
// Use single precision float. The first function should implement a serial map performed on the CPU;
// the second function should implement a parallel map in CUDA performed on the GPU.
// Check that the element-wise result on CPU is equal to the one computed on the GPU
// (modulo an epsilon error, e.g., fabs(cpu_res[i] - gpu-res[i]) < 0.0001 for all i),
// and print a VALID or INVALID message. Also make your program print:
// the runtimes of your CUDA vs CPU-sequential implementation
// (for CUDA please exclude the time for CPU-to-GPU transfer and GPU memory allocation),
// the acceleration speedup, the memory throughput (in GigaBytes/sec) of your CUDA implementation.
// Then increase the size of the array to determine what is roughly the maximal memory throughput.
// When you measure the GPU time:
// Call the CUDA kernel (repeatedly) inside a loop of some non-trivial count, say 300.
// After the loop, please place a cudaDeviceSynchronize(); statement.
// Measure the time before entering the loop and after the cudaDeviceSynchronize();
// --- the latter ensures that all Cuda kernels have actually finished execution ---
// then report the average kernel time, i.e., divide by the loop count.
// Please submit:
// your program named wa1-task3.cu together with a Makefile for it. your report, which should contain:
// whether it validates (and what epsilon have you used for validating the CPU to GPU results)
// the code of your CUDA kernel together with how it was called, including the code for the
// computation of the grid and block sizes. the array length for which you observed maximal throughput
// the memory throughput of your CUDA implementation (GB/sec) for that length and for the initial length
// (753411). In case you are not running on the dedicated servers,
// please also report the peak memory bandwidth of your GPU hardware.
// Important Observation: a very similar task is discussed in the slides of the first Lab, i.e.,
// the github folder HelperCode/Lab-1-Cuda contains a very naive CUDA implementation for multiplying
// with two each element of an array, but which works for arrays smaller than 1025 elements.
// The code in the slides generalizes the implementation to work correctly for arbitrary sizes.
// (The code in HelperCode/Lab-1-Cuda already has time instrumentation and validation;
// you may definitely take inspiration from there.)


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 100


// MAPPING FUNCTION \ x → (x/(x-2.3))^3
__global__ void mapKernel(float* X, float *Y, int N) {
    const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float x = X[gid];
    float partial_res = (x/(x-2.3));
    if (gid < N) {
        Y[gid] = partial_res*partial_res*partial_res;
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

      const unsigned int maxN = 500000000;
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

    unsigned int B = 256; // chose a suitable block size in dimension x
    unsigned int numblocks = (N + B -1) / B; // number of blocks in dimension x
    dim3 block(B,1,1), grid(numblocks,1,1); //totalnumberofthreads(numblocks*B)mayovershootN!
    // mul2Kernel<<<grid , block>>>(d in , d out,N); // pass N as parameter as well ;
    // d in and d out are in device memory

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        // mapKernel<<< 1, N>>>(d_in, d_out);
        mapKernel<<<grid , block>>>(d in , d out,N);
    }
  
    { // execute the kernel a number of times;
      // to measure performance use a large N, e.g., 200000000,
      // and increase GPU_RUNS to 100 or more. 
    
        double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int r = 0; r < GPU_RUNS; r++) {
            // mapKernel<<< 1, N>>>(d_in, d_out);
            mapKernel<<<grid , block>>>(d in , d out,N);
        }
        cudaDeviceSynchronize();
        // ^ `cudaDeviceSynchronize` is needed for runtime
        //     measurements, since CUDA kernels are executed
        //     asynchronously, i.e., the CPU does not wait
        //     for the kernel to finish.
        //   However, `cudaDeviceSynchronize` is expensive
        //     so we need to amortize it across many runs;
        //     hence, when measuring performance use a big
        //     N and increase GPU_RUNS to 100 or more.
        //   Sure, it would be better by using CUDA events, but
        //     the current procedure is simple & works well enough.
        //   Please note that the execution of multiple
        //     kernels in Cuda executes correctly without such
        //     explicit synchronization; we need this only for
        //     runtime measurement.
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", elapsed, gigabytespersec);
        
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    // print result
    //for(unsigned int i=0; i<N; ++i) printf("%.6f\n", h_out[i]);

    for(unsigned int i=0; i<N; ++i) {
        float actual   = h_out[i];
        // float expected = 2 * h_in[i]; 
        float x = h_in[i];
        float expected = pow((x/(x-2.3f)), 3.0); //\ x → (x/(x-2.3))^3
        if( abs(expected - actual) > 0.0001) { // (modulo an epsilon error, e.g., fabs(cpu_res[i] - gpu-res[i]) < 0.0001 for all i),
            printf("Invalid result at index %d, actual: %f, expected: %f. \n", i, actual, expected);
            exit(3);
        }
    }
    printf("Successful Validation.\n");

    // clean-up memory
    free(h_in);       free(h_out);
    cudaFree(d_in);   cudaFree(d_out);
}
