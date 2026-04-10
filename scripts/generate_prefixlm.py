import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_loader import ThunderModelLoader
from core.diffusion_model import PrefixLMDiffusionAdapter
from core.diffusion_engine import PrefixLMDiffusionEngine
from core.config_manager import THUNDER_CONFIG

def generate_text(prompt, steps=100, apply_clamping=True, checkpoint_steps=3000, guidance_scale=1.5):
    print(f"⚡ Thunder PrefixLM: Initializing Inference Engine...")
    
    loader = ThunderModelLoader()
    model, tokenizer = loader.load_model(load_in_4bit=THUNDER_CONFIG["hardware"]["load_in_4bit"])
    
    # Adapt model architecture first
    adapter = PrefixLMDiffusionAdapter(model)
    model = adapter.adapt_for_diffusion()
    
    # Updated path based on user's location
    checkpoint_path = f"/workspace/thunder_prefixlm_llama/checkpoint-{checkpoint_steps}"
    if os.path.exists(checkpoint_path):
        print(f"⚡ Loading trained weights from {checkpoint_path}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        adapter.load_diffusion_layers(checkpoint_path)
    else:
        # Try nested path just in case
        alt_path = f"/workspace/thunder/thunder_prefixlm_llama/checkpoint-{checkpoint_steps}"
        if os.path.exists(alt_path):
            print(f"⚡ Loading trained weights from {alt_path}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, alt_path, is_trainable=False)
            adapter.load_diffusion_layers(alt_path)
        else:
            print(f"⚠️  [WARNING] Checkpoint {checkpoint_path} not found. Standard initialization.")
    
    engine = PrefixLMDiffusionEngine(model)
    
    # 1. Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]
    
    # 2. Extract Clean Prompt Embeddings (Anchor) and Standardize
    emb_std = 0.02
    # The generation engine expects the embedding matrix to be standardized for the Clamping Trick
    embedding_matrix = model.get_input_embeddings().weight / emb_std
    
    prompt_embeds = model.get_input_embeddings()(input_ids) / emb_std
    
    # 3. Define the full sequence length (Prompt + Generation)
    generation_len = 32
    total_len = prompt_len + generation_len
    
    batch_size = 1
    hidden_size = model.config.hidden_size
    shape = (batch_size, total_len, hidden_size)
    
    # Pad prompt embeds to full shape (the engine will only anchor up to prompt_len)
    full_prompt_embeds = torch.zeros(shape, device=model.device, dtype=model.dtype)
    full_prompt_embeds[:, :prompt_len, :] = prompt_embeds
    
    # 4. Generate
    # Dynamic Temperature Schedule: High at start (discovery), Low at end (commitment)
    # steps=250: 0->200 (T=1.5), 200->250 (T=0.5)
    
    res = engine.generate(
        shape=shape,
        embedding_matrix=embedding_matrix,
        steps=steps,
        prompt_embeds=full_prompt_embeds,
        anchor_len=prompt_len,
        apply_clamping=apply_clamping,
        guidance_scale=guidance_scale,
        return_trajectory=True
    )
    
    _, final_tokens, trajectory = res
    
    # 5. Strict Persistence: Force the prompt part to be the ORIGINAL input_ids
    # This ensures "What is 5 and 7" NEVER becomes "What is 7 and 7"
    if prompt_len > 0:
        final_tokens[:, :prompt_len] = input_ids[:, :prompt_len]
    
    # 6. Decode
    generated_text = tokenizer.decode(final_tokens[0], skip_special_tokens=True)
    
    # 7. Save Trajectory to JSON for Logic Evaluation (Kedro Viz style)
    import json
    diag_data = {
        "metadata": {
            "prompt": prompt,
            "final": generated_text,
            "checkpoint": checkpoint_steps,
            "steps": steps
        },
        "nodes": [] # For graph tools
    }
    
    for entry in trajectory:
        step_text = tokenizer.decode(entry["tokens"], skip_special_tokens=True)
        diag_data["nodes"].append({
            "id": f"step_{entry['step']}",
            "type": "diffusion_state",
            "content": step_text,
            "is_final": (entry["step"] == steps - 1)
        })
        
    log_name = f"logic_graph_{checkpoint_steps}.json"
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log_name)
    
    # Overwrite on first prompt, append on others? 
    # Let's just append for now but clear it at the start of __main__
    with open(log_path, "a") as f:
        f.write(json.dumps(diag_data) + "\n")

    print("\n" + "="*50)
    print(f"PROMPT:\n{prompt}")
    print(f"\nGENERATED:\n{generated_text}")
    print(f"\n[INFO] Diagnostic trajectory saved to: {log_name}")
    print("="*50 + "\n")

def get_suggested_steps(prompt):
    """
    Heuristic to decide initial steps based on prompt nature.
    """
    prompt_lower = prompt.lower()
    
    # Simple Math/Factual
    if any(k in prompt_lower for k in ["sum of", "capital of", "speed in", "what is"]):
        return 15
    # Coding/Logic
    if any(k in prompt_lower for k in ["function", "code", "if all", "logic", "socrates"]):
        return 30
    # Creative/Complex
    return 50

if __name__ == "__main__":
    test_prompts = [
        # Math & Logic
        "### User:\nWhat is the sum of 5 and 7?\n\n### Assistant:\n",
        "### User:\nA train travels 60 miles in 2 hours. What is its speed in mph?\n\n### Assistant:\n",
        "### User:\nIf all humans are mortal, and Socrates is human, then Socrates is...\n\n### Assistant:\n",
        
        # Coding
        "### User:\nWrite a python hello world function.\n\n### Assistant:\n",
        "### User:\nWrite a JS function to reverse a string.\n\n### Assistant:\n",
        
        # Factual & Creative
        "### User:\nWhat is the capital of France?\n\n### Assistant:\n",
        "### User:\nWrite a short poem about the moon.\n\n### Assistant:\n",
        "### User:\nThe quick brown fox jumps over the lazy dog.\n\n### Assistant:\n"
    ]
    
    # Clear logs
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logic_graph_3000.json")
    if os.path.exists(log_path):
        os.remove(log_path)
        
    # Using checkpoint 3000 as requested by the user
    for prompt in test_prompts:
        dynamic_steps = get_suggested_steps(prompt)
        generate_text(prompt, steps=dynamic_steps, checkpoint_steps=3000)

