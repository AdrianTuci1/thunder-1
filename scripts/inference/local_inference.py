#!/usr/bin/env python3
import os
import sys
import torch
import argparse
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_manager import THUNDER_CONFIG
from core.scratch_dllm import ThunderScratchDiffusionLM, ScratchDLMConfig
from training.noise_scheduler import ThunderNoiseScheduler
from transformers import AutoTokenizer
from core.storage import ObjectStorageManager

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
        x0_normed = torch.nn.functional.normalize(x0_pred, dim=-1)
        emb_normed = torch.nn.functional.normalize(embedding_matrix, dim=-1)
        
        logits = torch.matmul(x0_normed, emb_normed.t())
        logits = logits * 30.0 
        
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            generated_ids = torch.multinomial(probs[0], num_samples=1).squeeze(-1)
        else:
            generated_ids = torch.argmax(logits[0], dim=-1)
        
        if prefix_len > 0:
             generated_ids[:prefix_len] = torch.tensor(prefix_ids, device=self.device)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Local Inference for Thunder dLLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-9765", help="Checkpoint name or full path")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--prompt", type=str, default=None, help="Initial text to complete")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--r2-sync", action="store_true", help="Attempt to download from R2 if not found locally")
    
    args = parser.parse_args()

    # --- Path Resolution ---
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Check if it's a folder in the default runs directory
        potential_path = os.path.join("runs", args.checkpoint)
        if os.path.exists(potential_path):
            checkpoint_path = potential_path
        elif args.r2_sync:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}. Attempting R2 download...")
            storage = ObjectStorageManager(THUNDER_CONFIG)
            dest_path = os.path.join("runs", args.checkpoint)
            success = storage.download_checkpoint(args.checkpoint, dest_path)
            if success:
                checkpoint_path = dest_path
            else:
                print("❌ Failed to download from R2.")
                sys.exit(1)
        else:
            print(f"❌ Checkpoint '{args.checkpoint}' not found. Use --r2-sync to pull from cloud.")
            sys.exit(1)

    model_state_pt = os.path.join(checkpoint_path, "model_state.pt")
    if not os.path.exists(model_state_pt):
        print(f"❌ Could not find model_state.pt in {checkpoint_path}")
        sys.exit(1)

    # --- Loading ---
    print(f"⚡ Loading model from {checkpoint_path} on {args.device}...")
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
    model.load_state_dict(torch.load(model_state_pt, map_location=args.device, weights_only=True))
    model.to(args.device)
    model.eval()
    
    scheduler = ThunderNoiseScheduler()
    sampler = DiffusionSampler(model, tokenizer, scheduler)

    print("✅ Model loaded successfully.\n")

    # --- Inference ---
    def run_inference(text):
        print(f"\n--- Generating completion for: '{text}' ---")
        result = sampler.sample(
            prefix_text=text,
            max_length=args.max_len,
            num_inference_steps=args.steps,
            temperature=args.temp
        )
        print(f"RESULT: {result}\n")

    if args.prompt:
        run_inference(args.prompt)
    else:
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("Prompt >>> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                run_inference(user_input)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
