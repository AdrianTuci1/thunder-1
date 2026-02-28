#include <cuda_runtime.h>

/**
 * THUNDER ENGINE: Fused Diffusion Operations
 * Minimizes global memory traffic by combining multiple element-wise
 * operations. Optimized for full-sequence arrays.
 */

__global__ void fused_diffusion_kernel(float *__restrict__ data,
                                       const float *__restrict__ bias,
                                       int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  // Fused operations: Scale + Bias (No boundary smoothing logic required)
  float val = data[i];
  val = (val * 1.05f) + bias[i];

  data[i] = val;
}

void launch_fused_diffusion(float *data, const float *bias, int size,
                            cudaStream_t stream) {
  dim3 threads(256);
  dim3 blocks((size + threads.x - 1) / threads.x);

  fused_diffusion_kernel<<<blocks, threads, 0, stream>>>(data, bias, size);
}
