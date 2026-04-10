⚡ Thunder | From-Scratch Bidirectional Diffusion-LM (dLLM)
============================================================

Thunder is a high-performance research implementation of a **Bidirectional Diffusion Transformer**, engineered from the ground up for massive throughput and complex sequence generation. By treating text generation as a parallel crystallization process rather than a token-by-token sequence, Thunder shatters the latency barriers of traditional LLMs.

### 📄 [Technical Deep Dive (Paper.md)](file:///Users/adriantucicovenco/Proiecte/thunder/paper.md)

---

## 🚀 Technical Specifications

| Feature | Specification |
| :--- | :--- |
| **Parameter Count** | ~817 Million (1B Architecture) |
| **Context Window** | 2048 Tokens (RoPE Scalable to 16k) |
| **Architecture** | Bidirectional Diffusion Transformer |
| **Positioning** | Rotary Positional Embeddings (RoPE) |
| **Optimization** | FlashAttention-2 & SwiGLU |
| **Data Mix** | FineWeb-Edu, Cosmopedia-v2, OpenWebMath, CodeParrot |

---

## 🧠 Core Architecture: Parallel Crystallization

Traditional LLMs are limited by sequential bottlenecks. Thunder treats the entire output space as a continuous latent field. 

*   **Non-Causal Transformer**: Every token attends to every other token, allowing the model to "think" globally across the entire context window.
*   **Latent Bridge**: Operates in a compressed latent space (1280 dim) while maintaining fidelity through high-dimensional token embeddings (1536 dim).
*   **Mercury-Style Budgets**: Multi-mode inference supporting:
    *   **Fast**: 5-10 denoising steps (Instant generation).
    *   **Thinking**: 25-50 steps (High coherence / Reasoning).

---

## 🛠️ Infrastructure & MLOps

Thunder is designed for hybrid scale: from **RTX 4090** local inference to **Multi-GPU A100/L40S** cloud training.

### dLLM Training Readiness Kit
<details>
<summary><b>📂 Configuration & Blueprints</b></summary>

- [blueprint.json](file:///Users/adriantucicovenco/Proiecte/thunder/configs/dllm_1b_blueprint.json) - Core model specification.
- [config_manager.py](file:///Users/adriantucicovenco/Proiecte/thunder/core/config_manager.py) - Training & Pipeline orchestration.
</details>

<details>
<summary><b>📂 Dataset & Integrity Scripts</b></summary>

- [verify_dataset.py](file:///Users/adriantucicovenco/Proiecte/thunder/scripts/verify_dataset_integrity.py) - Data quality validation.
- [verify_hf_sources.py](file:///Users/adriantucicovenco/Proiecte/thunder/scripts/verify_hf_dataset_sources.py) - Hub connectivity audit.
</details>

<details>
<summary><b>📂 Deployment & Training</b></summary>

- [launch_torchrun.sh](file:///Users/adriantucicovenco/Proiecte/thunder/scripts/launch_train_torchrun.sh) - Cluster initialization.
- [modal_train.py](file:///Users/adriantucicovenco/Proiecte/thunder/scripts/modal_train.py) - Serverless GPU training entrypoint.
- [run_from_scratch.py](file:///Users/adriantucicovenco/Proiecte/thunder/training/run_from_scratch.py) - Core training loop.
</details>

---

© 2026 [staticlabs.ro](https://staticlabs.ro). **Breaking the sequential barrier.**
