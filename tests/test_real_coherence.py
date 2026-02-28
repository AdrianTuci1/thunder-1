import sys
import os
import torch
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_loader import ThunderModelLoader
from core.diffusion_engine import ThunderDiffusionEngine
from core.scheduler import ThunderScheduler
from core.token_sampler import ThunderTokenSampler
from core.config_manager import THUNDER_CONFIG

def test_real_coherence():
    print("⚡ Thunder Real Output Test: Initialize...")
    
    # Load model and tokenizer
    loader = ThunderModelLoader()
    try:
        model, tokenizer = loader.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    scheduler = ThunderScheduler()
    engine = ThunderDiffusionEngine(model, scheduler)
    sampler = ThunderTokenSampler()
    sampler.temperature = 0.01 # Extremely sharp
    sampler.top_k = 1 # Greedy sampling
    
    # Use a sigmoid schedule which stays longer at low noise
    from training.noise_scheduler import ThunderNoiseScheduler
    engine.noise_scheduler = ThunderNoiseScheduler(schedule_type="sigmoid")
    
    # Test cases with proper template
    raw_prompts = [
        "What is the capital of France?",
        "Write a one-sentence greeting.",
    ]
    
    # Formatting for SFT-aligned model
    prompts = [f"### User:\n{p}\n\n### Assistant:\n" for p in raw_prompts]
    
    print(f"⚡ Testing {len(prompts)} prompts for real LLM output...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Input: {prompt.strip()}")
        
        # 1. Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_ids = inputs.input_ids
        
        # 2. Prepare latents: [Prompt Embeds] + [Noise Tokens for Response]
        num_response_tokens = 32 # Generate a short response
        
        with torch.no_grad():
            embed_module = model.get_input_embeddings()
            prompt_embeddings = embed_module(prompt_ids)
            
            # Prepare Unconditioned prompt for CFG
            null_prompt = "### User:\n\n### Assistant:\n"
            null_ids = tokenizer(null_prompt, return_tensors="pt").input_ids.to("cuda")
            null_embeddings = embed_module(null_ids)
            
            # Response placeholders (noise)
            # Scaling noise to match embedding magnitude (0.11)
            response_noise = torch.randn((1, num_response_tokens, prompt_embeddings.shape[-1]), device="cuda", dtype=prompt_embeddings.dtype) * 0.11
            
            # Full sequence field
            full_latents = torch.cat([prompt_embeddings, response_noise], dim=1)
            
            # 3. Run crystallization (Diffusion)
            start_time = time.time()
            # Increasing steps for higher fidelity + anchoring prompt part + CFG
            result_embeddings = engine.crystallize_sequence(
                full_latents, 
                steps=100, # 100 is enough with CFG
                anchor_len=prompt_ids.shape[1],
                guidance_scale=3.5,
                uncond_prompt_embeds=null_embeddings
            )
            end_time = time.time()
            
            # 4. Decode latent response to tokens
            # We only sample the response part
            response_latents = result_embeddings[:, prompt_ids.shape[1]:, :]
            
            # Project back to vocabulary using the model's lm_head
            sampled_ids = sampler.sample(response_latents, model.lm_head)
            
            # 5. Detokenize
            # sampled_ids is typically [B, L, 1] from sampler
            final_ids = sampled_ids.squeeze(-1)
            output_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
            
            print(f"Status: Success ✅")
            print(f"Response: {output_text.strip()}")
            print(f"Inference time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    test_real_coherence()
