1. Adaptarea folderului core/
Acesta este creierul modelului. Aici implementăm trecerea de la autoregresiv la bidirecțional.

diffusion_model.py: Aici trebuie să modifici clasa modelului (bazată pe Qwen) pentru a dezactiva causal masking. Trebuie să permiți self-attention-ului să fie global.

diffusion_engine.py: Aici implementezi bucla de denoising (cei 6 pași).

config_manager.py: Adaugă parametri pentru sigma_max, noise_schedule_type și setările de distilare.

2. Extinderea folderului kernels/
Pentru a "sparge" bariera pătratică, fișierele .cu (CUDA) sunt esențiale.

mercury_sculptor.cu: Acesta ar trebui să fie kernel-ul tău custom care optimizează adăugarea de zgomot Gaussian și pragul de decodare direct în SRAM-ul GPU-ului.

NOU: flash_attention_3.cu: Dacă vrei 32k context, ai nevoie de integrarea manuală a noilor kernel-uri FlashAttention care suportă măști non-cauzale eficiente.

3. Folderul training/ (Inima Distilării)
Aici se întâmplă magia "Teacher -> Student".

diffusion_lm_trainer.py: Trebuie adaptat pentru a încărca două modele simultan (Teacher Qwen-14B - frozen și Student 3B - active).

loss_functions.py: Adaugă Kullback–Leibler (KL) Divergence. Studentul nu trebuie doar să ghicească cuvântul, ci să imite distribuția de probabilitate a Teacher-ului.

noise_scheduler.py: Implementează aici curba de zgomot (Cosine sau Linear).

4. Folderul scripts/
NOU: convert_qwen_to_diffusion.py: Un script care ia modelul autoregresiv original și îi "injectează" noul DiffusionHead.

test_inference.py: Modifică-l pentru a măsura latența pe cei 6 pași vs. modul autoregresiv clasic.

5. Folderul tools/
context_shaper.py: Extrem de important pentru 32k. Trebuie să implementezi aici RoPE Scaling (Rotary Positional Embeddings) pentru a extinde capacitatea modelului de a înțelege distanțe mari între tokeni.

Structura Finală Sugerată (Modificări cu roșu):
Plaintext
thunder/
├── core/
│   ├── attention_handler.py    <-- (NOU) Gestionează masca bidirecțională
│   ├── diffusion_engine.py     <-- Loop-ul de 6 pași
│   └── distillation_wrapper.py <-- (NOU) Coordonează Teacher & Student
├── kernels/
│   ├── mercury_sculptor.cu
│   └── flash_attn_v3_interface.cu
├── training/
│   ├── distillation_logic.py   <-- (NOU) Implementează KL Divergence
│   ├── noise_scheduler.py      <-- Cosine noise schedule
│   └── loss_functions.py       <-- MSE (pentru vectori) + KL (pentru logits)
└── scripts/
    ├── run_distillation.py     <-- Scriptul principal de lansare
    └── benchmark_32k.py        <-- Testează presiunea pe VRAM la context mare