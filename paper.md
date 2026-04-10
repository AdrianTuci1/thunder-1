# Thunder dLLM: Technical Report & Training Journal

> **Project Status**: Pilot Training Phase
> **Model Scale**: 1 Billion Parameters (0.8B - 1.0B)
> **Architecture**: Bidirectional Diffusion Transformer (Thunder-1)

## 1. Abstract
Thunder dLLM is an open-source research initiative aimed at training a large-scale bidirectional diffusion language model from scratch. Unlike traditional autoregressive models, Thunder utilizes non-causal attention and a continuous latent space to enable global sequence denoising and "thinking" steps during inference.

## 2. Model Architecture
- **Type**: Bidirectional Diffusion Transformer
- **Positional Encoding**: Rotary Positional Embeddings (RoPE) with $\theta = 100,000$ (optimized for 16k context).
- **Activations**: SwiGLU
- **Normalization**: RMSNorm
- **Optimization**: FlashAttention-2 support for $O(L^2)$ scaling.
- **Latent Space**: Compressed linear bridge from token embeddings (1536) to latent space (1280).

## 3. Training Infrastructure
- **Main Hardware**: NVIDIA L40S (Pilot) / Planned 8x A100 (Sprint).
- **Environment**: Modal (Serverless Compute).
- **Monitoring**: Weights & Biases (WandB).
- **Storage**: Modal Volume (Persistent Object Store).

## 4. Experiment Log

### 4.1 Pilot Run: "Thunder-1B-Pilot-RoPE"
- **Date**: 2026-04-09
- **Hardware**: 1x NVIDIA L40S (48GB)
- **Duration**: 3 Hours
- **Dataset Mix**: Lean Data (FineWeb-Edu)
- **Tokens/sec (Throughput)**: [ADĂUGAȚI AICI]
- **Final Train Loss**: [ADĂUGAȚI AICI]
- **Observations**: Validated RoPE stability and FlashAttention compatibility. No gradient expansion issues detected.

### 4.2 Full Training Sprint: "Thunder-1B-Final"
- **Target**: [PLANIFICAT]
- **Hardware**: 8x NVIDIA A100 80GB
- **Objective**: 100B+ Tokens
- **Estimated Cost**: $[ADĂUGAȚI AICI]

## 5. Key Metrics
| Metric | Value |
| :--- | :--- |
| **Model Parameters** | ~817 Million |
| **Context Window** | 2048 (Expandable to 16k) |
| **Vocab Size** | 49,152 (SmolLM Tokenizer) |
| **MFU (Model Flops Utilization)**| [VALOARE %] |

## 5. Efficiency & Hardware Benchmarking
A key objective of this research is to compare the training and inference efficiency of Bidirectional Diffusion vs. standard Autoregressive (AR) models.

### 5.1 Training Efficiency (MFU)
We utilize Model FLOPs Utilization (MFU) as the primary metric for training efficiency, calculated as:
$$MFU = \frac{6 \cdot P \cdot T}{t \cdot \text{Peak FLOPS}}$$
Where $P$ is parameter count, $T$ is tokens per batch, and $t$ is step time.

| Model Type | Hardware | Tokens/s/GPU | MFU (%) |
| :--- | :--- | :--- | :--- |
| **Thunder-1B** | L40S | [CIFRĂ] | [CIFRĂ] |
| **GPT-1B (Ref)** | L40S | ~140,000 | ~35-40% |

### 5.2 Inference Throughput
Comparison of generation speed for a block of 1024 tokens.
- **Autoregressive (1B)**: $O(N)$ - Sequential token generation.
- **Thunder Diffusion (1B)**: $O(T)$ - Parallel block denoising in $T$ steps.

## 6. Text Generation & Quality Evaluation
To assess the linguistic competence of Thunder-1B, we use a multi-dimensional evaluation framework.

### 6.1 Perplexity (PPL)
Measured on the `fineweb-edu` validation set to quantify the model's ability to model the language distribution.

### 6.2 Sampling Efficiency
Evaluation of text quality relative to the number of denoising steps ($T$).
- **Draft Mode ($T=10$)**: Prioritizes speed for real-time applications.
- **Normal Mode ($T=25$)**: Balanced quality and latency.
- **High Quality ($T=100$)**: Maximum coherence for long-form generation.

### 6.3 Mauve Score (Proposed)
We plan to compute the Mauve score to measure the distributional similarity between generated text and human-written text from the validation corpus.

## 7. Comparative Analysis: Thunder vs. Autoregressive Baselines
To position Thunder-1B in the current LLM landscape, we compare its performance against established autoregressive (AR) models like Microsoft's **Phi-1.5/2** and hypothetical **2026 SOTA 1B AR models**.

### 7.1 Throughput vs. Sequence Length
One of the primary differentiators is how generation time scales with output length.

| Model | Architecture | 512 Tokens Speed | 8192 Tokens Speed |
| :--- | :--- | :--- | :--- |
| **Phi-2 (2.7B)** | AR (Sequential) | ~150 tok/s | ~15-20 tok/s (KV cache overhead) |
| **Thunder-1B** | Diffusion (Parallel)| **Constant Steps** | **High Throughput** |
| **2026 AR-1B** | AR (Optimized) | ~300 tok/s | ~50 tok/s |

### 7.2 The "Bidirectional" Edge
While Phi-series models excel at coding and reasoning, they lack the ability to "backtrack" or globally refine a generation. Thunder's diffusion process allows for **Parallel Crystallization**, where the entire sequence is refined simultaneously, leading to better global coherence in long-form narratives.

### 7.3 Benchmarking Roadmap: Towards LLM Arena
Our goal is to position Thunder-1B within the human-preference landscape.

| Benchmark | Target Baseline (Phi-1.5 / TinyLlama) | Thunder-1B Target |
| :--- | :--- | :--- |
| **HellaSwag** | ~60-70% | 55%+ |
| **MMLU** | ~25-35% | 25%+ |
| **HumanEval** | ~30-40% | 20%+ (focused on Python) |
| **Arena ELO** | ~1000-1100 | Competitive with 1B-tier models |

## 8. Hardware & Speed Evaluation
[This section will be populated with MFU and TPS data from the current L40S run.]

## 9. Qualitative Progress
- **Step 1000**: [DESCRIERE GENERARE]
- **Step 5000**: [DESCRIERE GENERARE]
- **Final**: [DESCRIERE GENERARE]

---
*Created with ❤️ by Antigravity (Advanced Agentic Coding)*
