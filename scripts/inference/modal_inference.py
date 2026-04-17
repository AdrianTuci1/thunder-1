import os
import sys
import modal
from typing import List, Optional

# Project path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modal Image Setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "tokenizers>=0.19.0",
        "tqdm",
        "boto3",
        "botocore",
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/thunder",
        ignore=[".git", "**/__pycache__", ".venv", "runs/**"],
    )
)

volume = modal.Volume.from_name("thunder-checkpoints", create_if_missing=True)
app = modal.App("thunder-inference")

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def run_interactive_test(checkpoint_name: str, prompts: List[str], temperature: float = 0.8, steps: int = 50):
    import os
    import sys
    
    # Intram in folderul proiectului din container
    os.chdir("/thunder")
    sys.path.insert(0, "/thunder")
    
    import torch
    import torch.nn.functional as F
    from core.config_manager import THUNDER_CONFIG
    from core.scratch_dllm import ThunderScratchDiffusionLM, ScratchDLMConfig
    from training.noise_scheduler import ThunderNoiseScheduler
    from transformers import AutoTokenizer
    from core.storage import ObjectStorageManager
    
    # --- Check for Checkpoint existence and download if missing ---
    checkpoint_base = "/checkpoints"
    checkpoint_path = os.path.join(checkpoint_base, checkpoint_name)
    model_state_path = os.path.join(checkpoint_path, "model_state.pt")
    
    if not os.path.exists(model_state_path):
        print(f"⚠️ Checkpoint '{checkpoint_name}' not found locally in volume. Attempting R2 download...")
        storage = ObjectStorageManager(THUNDER_CONFIG)
        success = storage.download_checkpoint(checkpoint_name, checkpoint_path)
        if not success:
            raise RuntimeError(f"Could not download checkpoint {checkpoint_name} from R2.")
        # Commit changes to the volume manually if needed, 
        # but Modal Volume usually syncs on write-back if configured (or we just use it during this run)
        volume.commit()
    
    # --- Incarcare Model ---
    print(f"⚡ Loading Thunder from {checkpoint_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    dlm_config = ScratchDLMConfig(
        vocab_size=len(tokenizer),
        max_seq_len=THUNDER_CONFIG["model"]["max_seq_len"],
        embedding_dim=THUNDER_CONFIG["model"]["embedding_dim"],
        latent_dim=THUNDER_CONFIG["model"]["latent_dim"],
        num_layers=THUNDER_CONFIG["model"]["num_layers"],
        num_attention_heads=THUNDER_CONFIG["model"]["num_attention_heads"],
        num_kv_heads=THUNDER_CONFIG["model"]["num_kv_heads"],
        ffn_hidden_size=THUNDER_CONFIG["model"]["ffn_hidden_size"],
    )
    
    model = ThunderScratchDiffusionLM(dlm_config)
    model_state = os.path.join(checkpoint_path, "model_state.pt")
    model.load_state_dict(torch.load(model_state, map_location="cuda", weights_only=True))
    model.to("cuda")
    model.eval()
    
    # --- Definire Sampler ---
    class DiffusionSampler:
        def __init__(self, model, tokenizer, scheduler):
            self.model = model
            self.tokenizer = tokenizer
            self.scheduler = scheduler
            self.device = model.device
            self.steps = scheduler.diffusion_steps
            self.alphas_cumprod = scheduler.alphas_cumprod.to(self.device).float()

        @torch.no_grad()
        def sample(
            self, 
            prefix_text: str = "", 
            max_length: int = 64, 
            num_inference_steps: int = 50,
            temperature: float = 1.0,
            self_conditioning: bool = True
        ) -> str:
            self.model.eval()
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_ids)
            if prefix_len >= max_length:
                return prefix_text
                
            embedding_matrix = self.model.get_input_embeddings().weight
            latent_dim = embedding_matrix.shape[1]
            x_t = torch.randn((1, max_length, latent_dim), device=self.device)
            
            if prefix_len > 0:
                prefix_ids_tensor = torch.tensor([prefix_ids], device=self.device)
                prefix_embeds = self.model.get_input_embeddings()(prefix_ids_tensor)
                x_t[:, :prefix_len, :] = prefix_embeds
                
            attention_mask = torch.ones((1, max_length), device=self.device)
            all_timesteps = torch.linspace(self.steps - 1, 0, num_inference_steps).long().tolist()
            
            self_cond = None
            for i, t_val in enumerate(all_timesteps):
                t_tensor = torch.full((1,), t_val, device=self.device, dtype=torch.long)
                x0_pred = self.model.diffusion_forward(
                    x_t=x_t,
                    t=t_tensor,
                    attention_mask=attention_mask,
                    self_cond=self_cond if self_conditioning else None
                )
                
                if self_conditioning:
                    self_cond = x0_pred
                
                alpha_t = self.alphas_cumprod[t_val]
                if i + 1 < len(all_timesteps):
                    alpha_t_prev = self.alphas_cumprod[all_timesteps[i+1]]
                else:
                    alpha_t_prev = torch.tensor(1.0, device=self.device)

                eps = (x_t - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(torch.clamp(1 - alpha_t, min=1e-8))
                x_t = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(torch.clamp(1 - alpha_t_prev, min=1e-8)) * eps
                
                if prefix_len > 0:
                    x_t[:, :prefix_len, :] = prefix_embeds

            # --- High-Quality Decoding: Cosine Similarity & Scaling ---
            # Normalizam vectorii pentru a elimina discrepanta de magnitudine (std 0.01 vs 1.0)
            x0_normed = F.normalize(x0_pred, dim=-1)
            emb_normed = F.normalize(embedding_matrix, dim=-1)
            
            # Calculam logit-ii bazati pe proximitate unghiulara
            logits = torch.matmul(x0_normed, emb_normed.t())
            
            # Scalam logit-ii (20.0 - 50.0 este standard pentru a "ascuti" distributia)
            logits = logits * 30.0 
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                generated_ids = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
            else:
                generated_ids = torch.argmax(logits[0], dim=-1)
            
            if prefix_len > 0:
                generated_ids[:prefix_len] = torch.tensor(prefix_ids, device=self.device)

            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    
    model = ThunderScratchDiffusionLM(dlm_config)
    model_state = os.path.join(checkpoint_path, "model_state.pt")
    model.load_state_dict(torch.load(model_state, map_location="cuda", weights_only=True))
    model.to("cuda")
    model.eval()
    
    scheduler = ThunderNoiseScheduler()
    sampler = DiffusionSampler(model, tokenizer, scheduler)
    
    print("\n" + "="*50)
    print(f"RUNNING INFERENCE ON {checkpoint_name}")
    print("="*50)
    
    for prompt in prompts:
        print(f"\nPROMPT: '{prompt}'")
        for i in range(2):
            completion = sampler.sample(
                prefix_text=prompt, 
                max_length=64, 
                num_inference_steps=steps,
                temperature=temperature
            )
            print(f"  Result {i+1}: {completion}")

@app.local_entrypoint()
def main(checkpoint: str = "checkpoint-9765", temp: float = 0.8, steps: int = 50):
    prompts = [
        # --- Knowledge & Science ---
        "Newton's first law of motion states that",
        "The fundamental theorem of calculus is used to",
        "The structure of a DNA molecule is known as",
        
        # --- Python Coding ---
        "import torch\nimport torch.nn as nn\n# Define a simple MLP classifier with 3 hidden layers:",
        "def quicksort(arr):\n    \"\"\"Implementation of the quicksort algorithm in Python.\"\"\"",
        
        # --- SQL / DB ---
        "SELECT e.name, d.department_name FROM employees e JOIN departments d ON",
        "CREATE TABLE users (id SERIAL PRIMARY KEY, username TEXT UNIQUE, email TEXT,",
        
        # --- General / Creative ---
        "In a future where AI and humans collaborate seamlessly,",
        "The primary advantage of a diffusion language model compared to an autoregressive model is",
        "StaticLabs is a research lab focusing on"
    ]
    run_interactive_test.remote(checkpoint, prompts, temperature=temp, steps=steps)
