/*
 * Copyright 2011-2015 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 */ 
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <thread>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define COMPUTE_N 50000
#define THREADS 8
#define ITERATIONS 1000

#ifdef TRACER
extern void initTrace(void);
extern void finiTrace(void);
#endif

// Kernels
__global__ void 
VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

__global__ void 
VecSub(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] - B[i];
}

static void
do_pass(cudaStream_t stream)
{
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;
  size_t size = COMPUTE_N * sizeof(int);
  int threadsPerBlock = 256;
  int blocksPerGrid = 0;
  
  // Allocate input vectors h_A and h_B in host memory
  // don't bother to initialize
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);
  
  // Allocate vectors in device memory
  RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

  RUNTIME_API_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  RUNTIME_API_CALL(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
  VecSub<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
    
  RUNTIME_API_CALL(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

/*
  if (stream == 0)
    RUNTIME_API_CALL(cudaDeviceSynchronize());
  else
    RUNTIME_API_CALL(cudaStreamSynchronize(stream));
*/

  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void myfunc(int deviceNum, cudaStream_t streamCommon) {
  CUdevice device;  
  char deviceName[32];
  DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
  DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
  printf("Device Name: %s\n", deviceName);

  RUNTIME_API_CALL(cudaSetDevice(deviceNum));
  cudaStream_t stream0;
  RUNTIME_API_CALL(cudaStreamCreate(&stream0));

  for (int i = 0 ; i < ITERATIONS ; i++) {
      // do pass default stream
      do_pass(0);
      // do pass with user stream
      do_pass(stream0);
      // do pass with common stream
      cudaStream_t stream2;
      // do pass with temporary stream
      RUNTIME_API_CALL(cudaStreamCreate(&stream2));
      do_pass(stream2);
      RUNTIME_API_CALL(cudaStreamDestroy(stream2));
  }
  //cudaDeviceSynchronize();
  RUNTIME_API_CALL(cudaStreamDestroy(stream0));
}

int
main(int argc, char *argv[])
{
  int deviceNum = 0, devCount = 0;

  printf("cuInit()...\n");
  DRIVER_API_CALL(cuInit(0));
  
  printf("cuGetDeviceCount()...\n");
  RUNTIME_API_CALL(cudaGetDeviceCount(&devCount));

  cudaStream_t stream0;
  printf("cuSetDevice()...\n");
  RUNTIME_API_CALL(cudaSetDevice(deviceNum));
  printf("cuStreamCreate()...\n");
  RUNTIME_API_CALL(cudaStreamCreate(&stream0));

  printf("spawning threads...\n");
  std::vector<std::thread> threads;
  for (int index=0; index<THREADS; index++) {
    std::thread t(myfunc,deviceNum,stream0);
    threads.push_back(std::move(t));
  }

  printf("joining threads...\n");
  for (int index=0; index<THREADS; index++) {
    threads[index].join();
  }

  printf("Destrying main stream...\n");
  RUNTIME_API_CALL(cudaDeviceSynchronize());
  RUNTIME_API_CALL(cudaSetDevice(deviceNum));
  RUNTIME_API_CALL(cudaStreamDestroy(stream0));

  return 0;
}

