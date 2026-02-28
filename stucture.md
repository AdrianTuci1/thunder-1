📂 Proiect: Thunder diffusion
/
├── .vscode/                    # Remote-SSH settings & CUDA GDB Debugger
├── kernels/                    # THE THUNDER ENGINE (Low-level performance)
│   ├── bidirectional_scan.cu   # Kernel Mamba pentru procesare globală (Context 120k)
│   ├── parallel_denoiser.cu    # + Difuzie paralelă pe micro-plăci (Multi-stream)
│   ├── selective_scan_fast.cu  # Optimizare Ada Lovelace (RTX 4090)
│   └── fused_diffusion.cu      # Fuzionarea operațiilor pentru latență minimă
├── core/                       # SISTEMUL NERVOS (Inference & Logic)
│   ├── model_loader.py         # Încărcare Phi-4/Mamba (Unsloth, 4-bit, BF16)
│   ├── model_adapter.py        # Injectare LoRA și activare bilateralitate
│   ├── state_manager.py        # + Gestionează "Global State" Mamba pentru cei 120k
│   ├── tile_manager.py         # + Fractal Tiling (120k -> 16k Plăci -> 2k Micro-plăci)
│   ├── stream_orchestrator.py  # + Execuție asincronă pe CUDA Streams paralele
│   ├── boundary_fuser.py       # + Netezirea (Blending) marginilor între micro-plăci
│   ├── diffusion_engine.py     # Motorul de cristalizare "All-at-Once" per placă
│   ├── scheduler.py            # ADAPTIVE SCHEDULER (Decide pașii per micro-placă)
│   ├── token_sampler.py        # Tranziția de la glitch/zgomot la text clar
│   └── visualizer.py           # Streamer asincron (Mapare probabilități -> Glitch)
├── reasoning/                  # CREIERUL (Intent & Personality)
│   ├── router.py               # Smart Gating (Internal vs Web Search)
│   ├── intent_analyzer.py      # + Calculează densitatea de calcul per tile ierarhic
│   └── personality.py          # Aliniere stil Gemini (System Prompt & Framing)
├── training/                   # HIGH-PERFORMANCE FINE-TUNING
│   ├── finetune_gemini.py      # Script principal SFT (Bilateral Denoising)
│   ├── noise_scheduler.py      # Controlul degradării datelor (Curba de zgomot)
│   ├── loss_functions.py       # Hybrid Loss (Denoising + Boundary Coherence)
│   ├── data_pipeline.py        # Constant Length Packing (Scaling până la 120k)
│   └── lora_config.py          # Optimizări Rank & Alpha pentru difuzie
├── tools/                      # UNELTE EXTERNE
│   ├── search_agent.py         # RAG asincron pentru context masiv
│   └── context_shaper.py       # Segmentarea datelor externe în structura ierarhică
├── configs/                    # CONFIGURAȚII DINAMICE
│   ├── hardware_4090.yaml      # Memory mapping, Stream count, Tensor Core limits
│   └── adaptive_rules.yaml     # Praguri de complexitate (120k auto-segmentation)
├── data/                       # STOCARE DATE PROCESATE
│   └── cache/                  # Tokeni procesați pentru antrenament fractal
├── app.py                      # Interfață WebSocket (Parallel Stream Display)
└── setup_env.sh                # Automatizare mediu (CUDA 12.x, Unsloth, SSM)