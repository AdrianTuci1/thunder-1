import torch
from transformers import AutoTokenizer

def analyze_embedding_space(embedding_path="thunder_prefixlm_llama/checkpoint-200/custom_embeddings.pt", model_path="unsloth/Llama-3.2-3B-Instruct"):
    """
    Analyzes the Euclidean and Cosine distances between Math concepts and Natural Language concepts
    in the learned Token Embedding matrix after End-to-End Diffusion-LM training.
    """
    try:
        embeddings = torch.load(embedding_path, map_location="cpu")
        print(f"✅ Loaded custom embeddings from {embedding_path}")
    except FileNotFoundError:
        print(f"❌ Could not find {embedding_path}. Ensure you have run some training steps.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Define our test sets
    math_tokens = ["axfrac", "boxed", "integral", "equation", "theorem", "math"]
    text_tokens = ["poem", "moon", "beautiful", "restaurant", "food", "friendly"]
    
    # Get token IDs
    math_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in math_tokens]
    text_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in text_tokens]
    
    print("\n--- Math Tokens ---")
    for t, tid in zip(math_tokens, math_ids):
        print(f"Token: '{t}' -> ID: {tid}")
        
    print("\n--- Text Tokens ---")
    for t, tid in zip(text_tokens, text_ids):
        print(f"Token: '{t}' -> ID: {tid}")

    # Extract the vectors
    math_vectors = embeddings[math_ids] # [len(math), H]
    text_vectors = embeddings[text_ids] # [len(text), H]
    
    # 1. In-Group Distances (How close are math tokens to each other?)
    math_center = math_vectors.mean(dim=0)
    text_center = text_vectors.mean(dim=0)
    
    # 2. Cross-Group Distance
    euclidean_dist = torch.norm(math_center - text_center).item()
    cosine_sim = torch.nn.functional.cosine_similarity(math_center.unsqueeze(0), text_center.unsqueeze(0)).item()
    
    print(f"\n--- Metrics ---")
    print(f"Distance between Math Center and Text Center (Euclidean): {euclidean_dist:.4f}")
    print(f"Cosine Similarity between Math Center and Text Center: {cosine_sim:.4f}")
    
    print("\n💡 INTERPRETATION:")
    print("If the embedding matrix is successfully learning from the Diffusion objective,")
    print("the Cosine Similarity between distinct conceptual clusters (Math vs Poetry) should decrease")
    print("over time as they are pushed apart into different regions of the N(0,1) latent space.")

if __name__ == "__main__":
    analyze_embedding_space()
