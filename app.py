import asyncio
from fastapi import FastAPI
from core.model_loader import ThunderModelLoader
from core.diffusion_engine import ThunderDiffusionEngine
from core.scheduler import ThunderScheduler
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

    async def initialize(self):
        print("⚡ Thunder: Booting engine...")
        model, tokenizer = self.loader.load_model()
        self.engine = ThunderDiffusionEngine(model, self.scheduler)
        print("⚡ Thunder: System Online.")

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
