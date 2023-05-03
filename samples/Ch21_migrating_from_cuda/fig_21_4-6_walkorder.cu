#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <iostream>

constexpr int count = 16 * 2;

__device__ int get_global_linear_id() {
  auto blockId = gridDim.x * blockIdx.y + blockIdx.x;
  return blockId * blockDim.y * blockDim.x +
    threadIdx.y * blockDim.x +
    threadIdx.x;
}

// BEGIN CODE SNIP #1, Part 1/2
__global__ void ExchangeKernel(int* dst) {
  auto index = get_global_linear_id(); // helper function
  auto fastest = threadIdx.x;
  auto neighbor = __shfl_xor_sync(0xFFFFFFFF, fastest, 1);
  dst[index] = neighbor;
}
// END CODE SNIP #1, Part 1/2

// BEGIN CODE SNIP #2
__global__ void ExchangeKernelCoopGroups(int* dst) {
  namespace cg = cooperative_groups;
  auto index = cg::this_grid().thread_rank();
  auto fastest = threadIdx.x;
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  auto neighbor = warp.shfl_xor(fastest, 1);
  dst[index] = neighbor;
}
// END CODE SNIP #2

int main() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "Running on device: " << deviceProp.name << "\n";

  int* buffer = nullptr;
  cudaMallocHost(&buffer, count * sizeof(int));
  cudaMemset(buffer, 0, count * sizeof(int));

#if 0
  // BEGIN CODE SNIP #1, Part 2/2
  dim3 threadsPerBlock(16, 2);
  ExchangeKernel<<<1, threadsPerBlock>>>(buffer);
  cudaDeviceSynchronize();
  // END CODE SNIP #1, Part 2/2
#else
  dim3 threadsPerBlock(16, 2);
  ExchangeKernelCoopGroups<<<1, threadsPerBlock>>>(buffer);
  cudaDeviceSynchronize();
#endif

  int unexpected = 0;
  for (int i = 0; i < count; i+=2) {
    if (buffer[i] == buffer[i+1]) {
      unexpected++;
    }
  }
  if (unexpected) {
    std::cout << "Error, found " << unexpected << " matching pairs.\n";
  } else {
    std::cout << "Success.\n";
  }

  cudaFreeHost(buffer);
  return 0;
}
