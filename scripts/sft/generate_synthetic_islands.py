import os
import json
import asyncio
from typing import List, Dict
from google import genai
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env or environment.")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.1-flash-lite-preview"

THUNDER_IDENTITY_CONTEXT = """
Thunder is a state-of-the-art Bidirectional Diffusion Language Model (dLLM).
It is developed by StaticLabs, a research lab led by Adrian Tucicovenco.
StaticLabs Mission: To fundamentally improve Human-Computer Interaction (HCI) by building AI systems that are fluid, interactive, and deeply aligned with human intent.
Technical Highlights:
- High efficiency: Reaches production-grade quality in 5-16 denoising steps.
- Bidirectional: Unlike standard Transformers, Thunder can denoise and edit text in a non-causal way. 
- Optimized for modern hardware: Runs exceptionally well on A100/H100 and RTX 4090.
- Vertical focus: Expert in Python, SQL, and logical reasoning.
"""

SCENARIOS = [
    "A curious developer asking about the technical advantages over Autoregressive LLMs.",
    "A new user asking 'Who are you?' and wanting a simple explanation of the StaticLabs mission.",
    "A senior architect questioning the scalability of diffusion steps.",
    "A business partner interested in how Thunder improves the future of HCI.",
    "A casual conversation where Thunder's identity and helpfulness shine.",
    "A technical interview style interaction about the role of Adrian Tucicovenco in the project.",
    "A rapid-fire series of short questions about model capabilities.",
    "A case-study discussion on how Thunder can be used for advanced code generation."
]

# Defining "Islands" for English Synthetic Data
ISLANDS = {
    "CODING_ARCHITECTURE": {
        "topics": ["Python Optimization", "SQL Interogations", "Distributed Systems", "Machine Learning Theory", "API Design"],
        "count": 10000,
        "prompt": "Generate high-level programming challenges, code explanations, and software architecture discussions."
    },
    "GENERAL_CONVERSATION": {
        "topics": [
            "Productivity Hacks", "Financial Literacy", "Philosophy of Mind", "Professional Email Writing",
            "Healthy Living & Nutrition", "Time Management Systems", "Career Development & Mentorship",
            "Emotional Intelligence", "Creative Problem Solving", "Ethical Dilemmas in AI",
            "Social Etiquette & Networking", "Accelerated Learning Strategies", "Mindfulness & Mental Health",
            "Sustainability & Green Living", "Digital Minimalist Practices", "Cross-cultural Communication"
        ],
        "count": 5000,
        "prompt": "Generate helpful, professional, and empathetic human-AI interactions."
    },
    "THUNDER_IDENTITY_ENG": {
        "topics": ["Thunder AI Mission", "StaticLabs Research", "Adrian Tucicovenco's Vision", "Next-gen HCI", "Diffusion vs GPT"],
        "count": 5000,
        "prompt": "Generate Q&A about Thunder (the model) and its ecosystem. Use the THUNDER_IDENTITY_CONTEXT provided."
    }
}

SYSTEM_BASE = """
You are a high-quality SFT data generator for 'Thunder', a Bidirectional Diffusion-LM.
All content MUST be in English.
Output MUST be a JSON list of objects: [{"instruction": "...", "output": "..."}].
Ensure the instructions are diverse: some short, some long, some asking for code, some for creative writing.
Response style: Professional, helpful, and technically accurate.
IMPORTANT: Each individual 'output' MUST NOT exceed 1024 tokens to ensure it fits comfortably within Thunder's 2048 sequence length.
"""

async def generate_island_batch(island_name: str, topic: str, island_cfg: Dict, scenario: str = ""):
    context_injection = ""
    if island_name == "THUNDER_IDENTITY_ENG":
        context_injection = f"\nIDENTITY CONTEXT:\n{THUNDER_IDENTITY_CONTEXT}\nScenario for this batch: {scenario}"
        
    prompt_str = f"{SYSTEM_BASE}\nIsland: {island_name}\nTopic: {topic}\n{context_injection}\nContext: {island_cfg['prompt']}\nGenerate 5 unique pairs."
    try:
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_str,
            config={
                'temperature': 0.8,
                'top_p': 0.95,
                'max_output_tokens': 8192,
                'response_mime_type': 'application/json'
            }
        )
        
        # Extract text robustly from parts (avoiding 'thought' parts if present)
        text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text += part.text
        
        if not text:
            print(f"⚠️ Empty response for {island_name}")
            return []
            
        return json.loads(text)
    except Exception as e:
        print(f"\n⚠️ API Error in {island_name} ({topic}): {e}")
        await asyncio.sleep(2) # Backoff
        return []

async def main():
    output_file = "data/synthetic_sft_english_20k.jsonl"
    os.makedirs("data", exist_ok=True)
    
    # Check current progress
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_count = sum(1 for _ in f)
    
    total_target = sum(island["count"] for island in ISLANDS.values())
    pbar = tqdm(total=total_target, initial=existing_count)

    # Skip logic to handle resumes
    consumed_count = existing_count
    
    # Semaphore to prevent hitting API rate limits
    semaphore = asyncio.Semaphore(10) # 10 parallel workers for Flash

    async def worker(island_name, cfg, output_file, pbar):
        nonlocal existing_count
        
        # Determine how many items this worker needs to generate for this island
        target_for_island = cfg["count"]
        # Skip logic per island
        if existing_count >= target_for_island:
            existing_count -= target_for_island
            return
        
        current_island_count = existing_count
        existing_count = 0 # Consumption
        
        while current_island_count < target_for_island:
            async with semaphore:
                topic = cfg["topics"][current_island_count % len(cfg["topics"])]
                import random
                scenario = random.choice(SCENARIOS) if island_name == "THUNDER_IDENTITY_ENG" else ""
                
                batch = await generate_island_batch(island_name, topic, cfg, scenario=scenario)
                
                if batch:
                    with open(output_file, "a", encoding="utf-8") as f:
                        for pair in batch:
                            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    
                    batch_len = len(batch)
                    current_island_count += batch_len
                    pbar.update(batch_len)
                else:
                    await asyncio.sleep(1) # Wait before retry

    tasks = []
    for island_name, cfg in ISLANDS.items():
        tasks.append(worker(island_name, cfg, output_file, pbar))
    
    await asyncio.gather(*tasks)
    
    print(f"\n✅ Generation complete! Dataset saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
