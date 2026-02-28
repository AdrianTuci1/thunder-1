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
    print(f"⚡ Thunder: Testing inference with query: '{query}'")
    
    # 1. Load Model
    loader = ThunderModelLoader()
    model, tokenizer = loader.load_model()
    
    # 2. Setup Engine
    scheduler = ThunderScheduler()
    router = ThunderRouter()
    personality = ThunderPersonality()
    engine = ThunderDiffusionEngine(model, scheduler)
    
    # 3. Route and Schedule
    route = router.route_query(query, forced_mode=mode)
    gen_mode = route["mode"]
    predicted_len = route.get("predicted_length", 128)
    
    steps = scheduler.calculate_steps(mode=gen_mode, predicted_length=predicted_len)
    print(f"⚡ Thunder: Route determined - Mode: {gen_mode}, Steps: {steps}")
    
    # 4. Generate (Crystallization)
    # For testing, we'll create a dummy initial noise based on a reasonable length
    # In a real scenario, this would be embeddings from the prompt prefix
    # Here we simplify to show the engine works
    
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    prompt_embeds = model.get_input_embeddings()(inputs.input_ids)
    
    # Create initial noise field of target length
    # Note: Thunder uses a full sequence field. For this test, we expand the prompt
    target_seq_len = inputs.input_ids.shape[1] + predicted_len
    initial_noise = torch.randn((1, target_seq_len, model.config.hidden_size), device=model.device, dtype=model.dtype)
    
    # We "join" the prompt embeds as conditioning or prefix
    # Simplified: just run crystallization on the noise
    output_latents = engine.crystallize_sequence(initial_noise, steps=steps, prompt_embeds=prompt_embeds)
    
    # 5. Decode
    # The output latents are in the embedding space [B, L, D]
    # We find the nearest token for each embedding vector
    print("⚡ Thunder: Crystallization complete. Decoding latents...")
    
    # Simple nearest-neighbor decoding in embedding space
    embeddings = model.get_input_embeddings().weight # [Vocab, D]
    
    # Calculate cosine similarity or L2 distance
    # For speed in test, we use dot product as a proxy for nearest neighbor
    # (assuming embeddings are somewhat normalized)
    logits = torch.matmul(output_latents, embeddings.t()) # [B, L, Vocab]
    token_ids = torch.argmax(logits, dim=-1) # [B, L]
    
    # Filter out tokens from the original prompt to see the "generated" parts
    prompt_len = inputs.input_ids.shape[1]
    generated_ids = token_ids[0, prompt_len:]
    
    # Decode to text
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Apply personality formatting
    formatted = personality.apply_formatting(response_text)
    print(f"\nFinal Response:\n{formatted}")

if __name__ == "__main__":
    test_query = "Explica-mi cum functioneaza difuzia paralela in Thunder."
    run_test_inference(test_query, mode="fast")
