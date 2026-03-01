import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_loader import ThunderModelLoader
from core.diffusion_engine import PrefixLMDiffusionEngine
from core.config_manager import THUNDER_CONFIG

def run_zero_shot_inference():
    print("⚡ Starting Zero-Shot LLaDA Inference Test...")
    print("⚠️  Warning: Output will likely be incoherent as the x0_head is not yet fine-tuned.")
    
    # 1. Load Model & Tokenizer
    loader = ThunderModelLoader(model_name="GSAI-ML/LLaDA-8B-Instruct")
    model, tokenizer = loader.load_model(load_in_4bit=True)
    
    # 2. Setup Engine
    engine = PrefixLMDiffusionEngine(model)
    
    # 3. Prepare Prompt
    test_query = "### User:\nExplain the concept of gravity.\n\n### Assistant:\n"
    inputs = tokenizer(test_query, return_tensors="pt").to(model.device)
    
    # Standardize prompt embeddings (sqrt scaling)
    hidden_size = model.config.hidden_size
    with torch.no_grad():
        prompt_embeds = model.get_input_embeddings()(inputs.input_ids)
        prompt_embeds = prompt_embeds * (hidden_size ** 0.5)
        
    # 4. Generate
    # We'll use 25 steps and the new Speedy Mode (threshold decoding)
    batch_size = 1
    # Generate 128 new tokens after the prompt
    total_len = inputs.input_ids.shape[1] + 128
    shape = (batch_size, total_len, hidden_size)
    
    print("⚡ Thunder: Running generation loop...")
    
    embedding_matrix = model.get_input_embeddings().weight.data
    
    output_latents, output_token_ids = engine.generate(
        shape=shape,
        embedding_matrix=embedding_matrix,
        steps=25,
        prompt_embeds=prompt_embeds,
        anchor_len=inputs.input_ids.shape[1],
        threshold=0.05 # Speedy Mode enabled
    )
    
    # 5. Decode
    # Only decode the generated part (Assistant response)
    response_tokens = output_token_ids[0, inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    print("\n" + "="*50)
    print(f"QUERY: {test_query}")
    print("-"*50)
    print(f"ZERO-SHOT RESPONSE:\n{response_text}")
    print("="*50)

if __name__ == "__main__":
    run_zero_shot_inference()
