#include <cerrno>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define ull unsigned long long

template <typename T, typename P> T cceil(const T &a, const P &b) {
  return (a + b - 1) / b;
}

using namespace std;
__global__ void regular(char *p, ull size, ull iterations_per_block,
                        ull elements_per_thread) {
  ull tid = blockIdx.x * blockDim.x + threadIdx.x;
  ull start_idx = tid * elements_per_thread;
  ull end_idx = min(start_idx + elements_per_thread, size);

  // Each thread iterates 'iterations_per_block' times over its assigned block
  for (ull k = 0; k < iterations_per_block; ++k) {
    for (ull i = start_idx; i < end_idx; ++i) {
      p[i] = 'R' + (char)(i % 26);
    }
  }
}

__global__ void irregular(char *p, ull *offset_indices, ull size, ull iter) {
  ull tid = blockIdx.x * blockDim.x + threadIdx.x;
  ull current_off = tid % size;
  while (iter--) {
    p[current_off] = 'I';
    current_off = offset_indices[current_off];
    current_off %= size;
  }
}

void usage() {
  cout << "usage: ./uvm_work [size] [r/i] [seed] [iter] | grid block \n";
  cout << "size - TOTAL cudaMallocManaged size in GB\n";
  cout << "r/i - regular or irregular workload\n";
  cout << "seed - seed for random offset on irregular workload - ONLY ON "
          "IRREGULAR\n";
  cout << "iter - Number of iterations in a pattern\n";
  cout << "| - Optional start after this \n";
  cout << "grid block - used to launch the kernel. Default <4096,32>\n";
}

int main(int argc, char *argv[]) {
  char *devPtr;

  if (argc != 5 && argc != 7) {
    usage();
    return -EINVAL;
  }

  ull size = 0;
  const ull GB = 1ULL << 30;
  sscanf(argv[1], "%llu", &size);
  size *= GB;

  char mode = argv[2][0];
  if (mode != 'r' && mode != 'i') {
    usage();
    return -EINVAL;
  }

  ull seed = 0;
  sscanf(argv[3], "%llu", &seed);

  // ull iter = 0;
  // sscanf(argv[4], "%llu", &iter);

  ull grid = 4096, block = 32;
  if (argc > 5) {
    sscanf(argv[5], "%llu", &grid);
    sscanf(argv[6], "%llu", &block);
  }
  ull totalThreads = grid * block;

  if (mode == 'r') {
    ull perThreadHandle = cceil(size, grid * block);
    cout << "\nGrid :" << grid << "\nBlock: " << block
         << "\ntotalThreads: " << totalThreads
         << "\nperThread :" << perThreadHandle << "\n";
    cudaMallocManaged(&devPtr, size);
    regular<<<grid, block>>>(devPtr, size, perThreadHandle, perThreadHandle);
  } else {
    ull *randomNum;
    ull randStateSize = totalThreads * sizeof(ull);
    cudaMallocManaged(&randomNum, randStateSize);
    ull to_alloc = size - randStateSize;
    cudaMallocManaged(&devPtr, to_alloc);
    ull perThreadHandle = cceil(to_alloc, grid * block);
    cout << "\nGrid :" << grid << "\nBlock: " << block
         << "\ntotalThreads: " << totalThreads
         << "\nperThread :" << perThreadHandle << "\nSize: " << to_alloc
         << "\n";

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<ull> distrib(0, to_alloc - 1);
    for (int i = 0; i < totalThreads; ++i) {
      randomNum[i] = distrib(gen);
    }
    irregular<<<grid, block>>>(devPtr, randomNum, to_alloc, perThreadHandle);
    size = to_alloc;
  }

  cudaDeviceSynchronize();
  for (unsigned long long i = 0; i < size; i += 1024) {
    devPtr[i] += 2;
  }
  cout << "DONE\n";
}
