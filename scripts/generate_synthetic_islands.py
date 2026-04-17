import os
import json
import asyncio
from typing import List, Dict
import google.generativeai as genai
from tqdm.asyncio import tqdm

# Configuration
# Make sure to set your GEMINI_API_KEY as an environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Defining "Islands" for English Synthetic Data
ISLANDS = {
    "SCIENCE_HUMANITIES": {
        "topics": ["Quantum Physics", "World History", "Molecular Biology", "Space Exploration", "Psychology"],
        "count": 12000,
        "prompt": "Generate complex questions and detailed answers about science, technology, and human history."
    },
    "CODING_ARCHITECTURE": {
        "topics": ["Python Optimization", "SQL Interogations", "Distributed Systems", "Machine Learning Theory", "API Design"],
        "count": 15000,
        "prompt": "Generate high-level programming challenges, code explanations, and software architecture discussions."
    },
    "MATHEMATICAL_LOGIC": {
        "topics": ["Calculus", "Probability Theory", "Symbolic Logic", "Game Theory", "Paradoxes"],
        "count": 10000,
        "prompt": "Generate step-by-step mathematical proofs and logical reasoning puzzles."
    },
    "GENERAL_CONVERSATION": {
        "topics": ["Productivity Hacks", "Financial Literacy", "Philosophy of Mind", "Professional Email Writing"],
        "count": 8000,
        "prompt": "Generate helpful, professional, and empathetic human-AI interactions."
    },
    "THUNDER_IDENTITY_ENG": {
        "topics": ["Thunder AI Mission", "StaticLabs Research", "Adrian Tucicovenco's Vision"],
        "count": 5000,
        "prompt": "Generate Q&A about Thunder (the model), developed by StaticLabs led by Adrian Tucicovenco. Focus on the advantages of Diffusion-LMs."
    }
}

SYSTEM_BASE = """
You are a high-quality SFT data generator for 'Thunder', a Bidirectional Diffusion-LM.
All content MUST be in English.
Output MUST be a JSON list of objects: [{"instruction": "...", "output": "..."}].
Ensure the instructions are diverse: some short, some long, some asking for code, some for creative writing.
Response style: Professional, helpful, and technically accurate.
"""

async def generate_island_batch(island_name: str, topic: str, island_cfg: Dict):
    prompt_str = f"{SYSTEM_BASE}\nIsland: {island_name}\nTopic: {topic}\nContext: {island_cfg['prompt']}\nGenerate 10 unique pairs."
    try:
        response = await model.generate_content_async(
            prompt_str,
            generation_config=genai.GenerationConfig(temperature=0.8, top_p=0.95)
        )
        text = response.text.strip()
        
        # Strip potential markdown backticks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text)
    except Exception:
        return []

async def main():
    output_file = "data/synthetic_sft_english_50k.jsonl"
    os.makedirs("data", exist_ok=True)
    
    # Check current progress
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_count = sum(1 for _ in f)
    
    total_target = sum(island["count"] for island in ISLANDS.values())
    pbar = tqdm(total=total_target, initial=existing_count)

    # Semaphore to prevent hitting API rate limits
    async with asyncio.Semaphore(10): 
        for island_name, cfg in ISLANDS.items():
            current_island_count = 0
            # Rough local count for the specific island within this run
            while current_island_count < cfg["count"]:
                topic = cfg["topics"][current_island_count % len(cfg["topics"])]
                batch = await generate_island_batch(island_name, topic, cfg)
                
                if batch:
                    with open(output_file, "a", encoding="utf-8") as f:
                        for pair in batch:
                            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    
                    batch_len = len(batch)
                    current_island_count += batch_len
                    pbar.update(batch_len)
                
                # Small delay to keep the API happy
                await asyncio.sleep(0.2)
    
    print(f"\n✅ Generation complete! Dataset saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
