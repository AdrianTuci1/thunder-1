import sys
import os
import torch
from unsloth import FastLanguageModel

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config_manager import THUNDER_CONFIG

def test_autoregressive_coherence():
    print("⚡ Thunder Autoregressive Fallback Test: Initialize...")
    
    model_path = THUNDER_CONFIG["engine"]["model_path"]
    print(f"⚡ Loading model from {model_path}...")
    
    # Load model and tokenizer via Unsloth (Standard way)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    # Test prompts
    prompts = [
        "Explain the process of nuclear fusion in the sun in about 500 words, being very detailed about the physics involved.",
        "Solve this logic puzzle: If Sally has 3 brothers, and each of those brothers has 2 sisters, how many sisters does Sally have?",
        "Write a Python function to find the most frequent element in a list, and explain how it works.",
        "Compare and contrast the philosophical ideas of Stoicism and Epicureanism regarding the pursuit of happiness.",
        "Write a short science fiction story about a planet where it never stops raining, but the rain is made of liquid diamonds."
    ]
    
    import torch._dynamo
    with torch._dynamo.config.patch(disable=True):
        for prompt in prompts:
            print(f"\n--- Prompt ---")
            print(f"Input: {prompt}")
            
            # Use Chat ML like format consistent with SFT data
            formatted_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
            
            inputs = tokenizer([formatted_prompt], return_tensors = "pt").to("cuda")
            
            # Generate using standard HF generate (bypassing our diffusion engine)
            outputs = model.generate(
                **inputs, 
                max_new_tokens = 1024, # Increased for long-form answers
                use_cache = True,
                temperature = 0.7,
                top_p = 0.9,
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the assistant part
            if "### Assistant:" in response:
                response = response.split("### Assistant:")[-1].strip()
                
            print(f"Response: {response}")

if __name__ == "__main__":
    test_autoregressive_coherence()
