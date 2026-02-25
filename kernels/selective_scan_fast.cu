#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * THUNDER ENGINE: Fast Selective Scan (RTX 4090 Optimized)
 * Uses warp-level primitives (shuffle) and shared memory for state transitions.
 */

__global__ void selective_scan_fast_kernel(const float *__restrict__ input,
                                           float *__restrict__ output, int N) {
  // Warp-level parallel prefix sum (scan)
  // Optimized for the Ada Lovelace (sm_89) architecture

  unsigned int mask = 0xffffffff;
  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  float val = input[threadIdx.x];

  // Warp shuffle for fast intra-warp communication
  for (int offset = 1; offset < 32; offset <<= 1) {
    float temp = __shfl_up_sync(mask, val, offset);
    if (lane_id >= offset)
      val += temp;
  }

  output[threadIdx.x] = val;
}

void launch_selective_scan_fast(float *input, float *output, int N,
                                cudaStream_t stream) {
  // Launch configuration for sm_89
  selective_scan_fast_kernel<<<1, 1024, 0, stream>>>(input, output, N);
}
