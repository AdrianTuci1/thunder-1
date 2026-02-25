#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>

namespace cg = cooperative_groups;

/**
 * THUNDER ENGINE: Bidirectional Mamba Scan Kernel
 * Optimized for Phi-4 120k context processing.
 *
 * Performs forward and backward SSM scans simultaneously.
 */

__global__ void bidirectional_mamba_scan_kernel(
    const float *__restrict__ u_ptr,     // Input: [B, L, D]
    const float *__restrict__ A_ptr,     // SSM A Matrix: [D] (Simplified)
    const float *__restrict__ B_ptr,     // SSM B Matrix: [D] (Simplified)
    const float *__restrict__ C_ptr,     // SSM C Matrix: [D] (Simplified)
    const float *__restrict__ delta_ptr, // Time delta: [B, L, D]
    float *__restrict__ y_ptr,           // Output: [B, L, D]
    int batch_size, int L, int D) {
  // Parallelizing along Batch (B) and Dimension (D)
  int b = blockIdx.y;
  int d = blockIdx.x * blockDim.x + threadIdx.x;

  if (d >= D)
    return;

  // Parameters for this dimension
  float A = A_ptr ? A_ptr[d] : -1.0f;
  float B = B_ptr ? B_ptr[d] : 1.0f;
  float C = C_ptr ? C_ptr[d] : 1.0f;

  // Memory pointers
  const float *u_b = u_ptr + b * L * D + d;
  const float *delta_b = delta_ptr + b * L * D + d;
  float *y_b = y_ptr + b * L * D + d;

  // Forward Pass
  float h_fw = 0.0f;
  for (int l = 0; l < L; ++l) {
    float u_val = u_b[l * D];
    float dt = delta_b[l * D];

    // Euler Discretization
    // A_bar = exp(dt * A)
    // B_bar = dt * B
    float A_bar = expf(dt * A);
    float B_bar = dt * B;

    h_fw = A_bar * h_fw + B_bar * u_val;
    y_b[l * D] = h_fw * C;
  }

  // Backward Pass
  float h_bw = 0.0f;
  for (int l = L - 1; l >= 0; --l) {
    float u_val = u_b[l * D];
    float dt = delta_b[l * D];

    float A_bar = expf(dt * A);
    float B_bar = dt * B;

    h_bw = A_bar * h_bw + B_bar * u_val;

    // Fuse backward result with forward result
    y_b[l * D] += h_bw * C;
  }
}

void launch_bidirectional_scan(const float *u, const float *A, const float *B,
                               const float *C, const float *delta, float *y,
                               int B_dim, int L, int D, cudaStream_t stream) {
  dim3 threads(256);
  dim3 blocks((D + threads.x - 1) / threads.x, B_dim);

  bidirectional_mamba_scan_kernel<<<blocks, threads, 0, stream>>>(
      u, A, B, C, delta, y, B_dim, L, D);
}
