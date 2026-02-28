import asyncio
from fastapi import FastAPI
from core.model_loader import ThunderModelLoader
from core.diffusion_engine import PrefixLMDiffusionEngine
from core.scheduler import ThunderScheduler
from core.dynamic_batching import DynamicBatcher
from reasoning.router import ThunderRouter
from reasoning.personality import ThunderPersonality
from core.config_manager import THUNDER_CONFIG

app = FastAPI(title="Thunder Inference Engine")

class ThunderApp:
    def __init__(self):
        self.loader = ThunderModelLoader()
        self.router = ThunderRouter()
        self.scheduler = ThunderScheduler()
        self.personality = ThunderPersonality()
        self.engine = None
        self.batcher = None

        model, tokenizer = self.loader.load_model()
        self.engine = PrefixLMDiffusionEngine(model)
        
        # Initialize the dynamic batcher for high-throughput inference (Mercury 1 Adaptation)
        # Now uses the refined adaptive scheduler for optimal steps calculation
        self.batcher = DynamicBatcher(self.engine, tokenizer, self.scheduler, max_batch_size=16, max_wait_ms=100)
        
        print("⚡ Thunder: System Online (Dynamic Batching & Paging Enabled).")

thunder = ThunderApp()

@app.on_event("startup")
async def startup_event():
    await thunder.initialize()
    
    # OpenAI API Interface Mounting
    from api.openai_api import setup_openai_routes
    app.include_router(setup_openai_routes(thunder))
    print("⚡ Thunder: OpenAI API Interface Enabled.")

if __name__ == "__main__":
    import uvicorn
    host = THUNDER_CONFIG["server"]["host"]
    port = THUNDER_CONFIG["server"]["port"]
    uvicorn.run(app, host=host, port=port)
