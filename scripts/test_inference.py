import os
import sys
import torch
from unsloth import FastLanguageModel

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import ThunderModelLoader
from core.diffusion_engine import ThunderDiffusionEngine
from core.scheduler import ThunderScheduler
from reasoning.router import ThunderRouter
from reasoning.personality import ThunderPersonality
from core.config_manager import THUNDER_CONFIG

def run_test_inference(query, mode=None):
    print("⚡ Thunder: Running inference test with Qwen3.5-9B Diffusion...")
    
    loader = ThunderModelLoader()
    model, tokenizer = loader.load_model()
    
    # 2. Setup Engine
    scheduler = ThunderScheduler()
    router = ThunderRouter()
    personality = ThunderPersonality()
    engine = ThunderDiffusionEngine(model, scheduler)
    
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    anchor_len = inputs.input_ids.shape[1]
    
    # 3. Route and Schedule
    route = router.route_query(query, forced_mode=mode)
    gen_mode = route["mode"]
    
    steps = scheduler.calculate_steps(mode=gen_mode, anchor_len=anchor_len)
    print(f"⚡ Thunder: Route determined - Mode: {gen_mode}, Steps: {steps}")
    
    # 4. Generate (Crystallization)
    # For testing, we'll create a dummy initial noise based on a reasonable length
    # In a real scenario, this would be embeddings from the prompt prefix
    # Here we simplify to show the engine works
    
    prompt_embeds = model.get_input_embeddings()(inputs.input_ids)
    
    # Create initial noise field of target length
    predicted_len = route.get("predicted_length", 128)
    target_seq_len = anchor_len + predicted_len
    initial_noise = torch.randn((1, target_seq_len, model.config.hidden_size), device=model.device, dtype=model.dtype)
    
    # Setup embeddings for generation
    embeddings = model.get_input_embeddings().weight.detach()
    
    # We "join" the prompt embeds as conditioning or prefix
    # Simplified: just run crystallization on the noise
    _, token_ids = engine.generate(
        shape=initial_noise.shape,
        embedding_matrix=embeddings,
        steps=steps,
        prompt_embeds=prompt_embeds,
        anchor_len=inputs.input_ids.shape[1],
        max_new_tokens=predicted_len # Dynamic Canvas scaling
    )
    
    # Simple nearest-neighbor decoding is already done by engine.generate's final clamping if we want, 
    # but test_inference.py tries to do it manually. Let's use the final_tokens from generate.
    
    # 5. Decode
    # The output latents are in the embedding space [B, L, D]
    # We find the nearest token for each embedding vector
    print("⚡ Thunder: Crystallization complete. Decoding latents...")
    
    generated_ids = token_ids[0, anchor_len:]
    
    # Decode to text
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Apply personality formatting
    formatted = personality.apply_formatting(response_text)
    print(f"\nFinal Response:\n{formatted}")

if __name__ == "__main__":
    test_query = "Explica-mi cum functioneaza difuzia paralela in Thunder."
    run_test_inference(test_query, mode="fast")
