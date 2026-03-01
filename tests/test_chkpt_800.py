import os
import sys
import torch
from unsloth import FastLanguageModel

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from core.model_loader import ThunderModelLoader
from core.diffusion_engine import PrefixLMDiffusionEngine
from core.scheduler import ThunderScheduler
from reasoning.router import ThunderRouter
from reasoning.personality import ThunderPersonality
from core.config_manager import THUNDER_CONFIG

def run_test_inference(query, checkpoint_path, mode=None):
    print(f"⚡ Thunder: Testing inference on checkpoint: {checkpoint_path}")
    
    # 1. Format prompt based on training data discovery
    formatted_query = f"### User:\n{query}\n\n### Assistant:\n"
    print(f"⚡ Thunder: Formatted Query:\n{formatted_query}")
    
    # 2. Load Model
    loader = ThunderModelLoader(model_name=checkpoint_path)
    model, tokenizer = loader.load_model()
    
    # 3. Setup Engine
    scheduler = ThunderScheduler()
    router = ThunderRouter()
    personality = ThunderPersonality()
    engine = PrefixLMDiffusionEngine(model)
    
    # 4. Route and Schedule
    route = router.route_query(query, forced_mode=mode)
    gen_mode = route["mode"]
    predicted_len = route.get("predicted_length", 128)
    
    # Increase steps for better convergence at inference
    steps = 100 
    print(f"⚡ Thunder: Route determined - Mode: {gen_mode}, Steps: {steps}")
    
    # 5. Generate
    inputs = tokenizer(formatted_query, return_tensors="pt").to(model.device)
    raw_prompt_embeds = model.get_input_embeddings()(inputs.input_ids)
    
    # Scaling factor for standardized space (sqrt(hidden_size))
    emb_scale = (model.config.hidden_size ** 0.5)
    prompt_embeds = raw_prompt_embeds * emb_scale
    
    print(f"⚡ Debug: Prompt Latent Scale (mean abs): {prompt_embeds.abs().mean().item():.4f}")
    
    # Grounding Fix: Anchor length is the prompt length
    anchor_len = inputs.input_ids.shape[1]
    target_seq_len = anchor_len + predicted_len
    shape = (1, target_seq_len, model.config.hidden_size)
    
    # Get raw embedding matrix for clamping
    embedding_matrix = model.get_input_embeddings().weight
    
    print("⚡ Thunder: Starting Generation Loop...")
    latents, token_ids = engine.generate(
        shape=shape,
        embedding_matrix=embedding_matrix,
        steps=steps,
        prompt_embeds=prompt_embeds,
        anchor_len=anchor_len,
        guidance_scale=1.5,
        apply_clamping=True
    )
    
    # 6. Decode
    print("⚡ Thunder: Generation complete. Decoding tokens...")
    
    # Filter out tokens from the original prompt to see the "generated" parts
    generated_ids = token_ids[0, anchor_len:]
    
    # Decode to text
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Apply personality formatting
    formatted = personality.apply_formatting(response_text)
    print(f"\nFinal Response:\n{formatted}")

if __name__ == "__main__":
    checkpoint = os.path.join(PROJECT_ROOT, "thunder_prefixlm_llama/checkpoint-800")
    test_queries = [
        "Explain how diffusion works.",
        "What is the capital of France?",
        "Write a Python function to sort a list.",
        "How is Thunder different from standard LLMs?"
    ]
    for query in test_queries:
        print("\n" + "="*50)
        run_test_inference(query, checkpoint, mode="fast")
        print("="*50 + "\n")
