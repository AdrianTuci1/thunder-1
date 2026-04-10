#include <cuda_runtime.h>

/**
 * MERCURY 1: Parallel Sequence Sculpting Kernel (Paged)
 * Optimized for RTX 4090. Handles simultaneous denoising across physical pages.
 */

__global__ void mercury_sculpt_kernel(const float *__restrict__ input_noise,
                                      const float *__restrict__ predicted_x0,
                                      float *__restrict__ output_data,
                                      const int *__restrict__ page_table,
                                      int page_size, int hidden_size,
                                      float sqrt_alpha_next,
                                      float sqrt_one_minus_alpha_next,
                                      float sqrt_alpha_now, int total_tokens) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= total_tokens)
    return;

  // Address Translation for Paged Latents
  int page_idx = token_idx / page_size;
  int offset = token_idx % page_size;
  int phys_page = page_table[page_idx];

  int base_offset = (phys_page * page_size + offset) * hidden_size;

  for (int d = 0; d < hidden_size; d++) {
    int idx = base_offset + d;
    float xt = input_noise[idx];
    float x0 = predicted_x0[idx];

    // Mercury Sculpting Logic: x_{t-1} = sqrt(alpha_next) * x_0 +
    // sqrt(1-alpha_next) * noise Where noise = (x_t - sqrt(alpha_now)*x_0) /
    // sqrt(1-alpha_now)
    float noise_est = (xt - sqrt_alpha_now * x0);
    output_data[idx] =
        sqrt_alpha_next * x0 + sqrt_one_minus_alpha_next * noise_est;
  }
}

void launch_mercury_sculpt(float *input, float *predicted_x0, float *output,
                           const int *page_table, int page_size,
                           int hidden_size, int total_tokens,
                           float sqrt_alpha_next,
                           float sqrt_one_minus_alpha_next,
                           float sqrt_alpha_now, cudaStream_t stream) {
  dim3 threads(64);
  dim3 blocks((total_tokens + threads.x - 1) / threads.x);
  mercury_sculpt_kernel<<<blocks, threads, 0, stream>>>(
      input, predicted_x0, output, page_table, page_size, hidden_size,
      sqrt_alpha_next, sqrt_one_minus_alpha_next, sqrt_alpha_now, total_tokens);
}
