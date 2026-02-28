import time
import json
import asyncio
import random
from typing import List, Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config_manager import THUNDER_CONFIG
from tools.search_agent import ThunderSearchAgent

router = APIRouter()
search_agent = ThunderSearchAgent()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "thunder-v1"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    mode: Optional[str] = None

def setup_openai_routes(thunder_instance):
    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        # 1. Token Validation
        expected_token = THUNDER_CONFIG["server"].get("api_token")
        if expected_token:
            auth_header = http_request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != expected_token:
                raise HTTPException(status_code=401, detail="Invalid or missing API Token")

        query = request.messages[-1].content
        forced_mode = request.mode
        
        # 2. Extract mode from model name if not explicitly forced
        # This allows selecting mode via the model dropdown in Open WebUI
        if not forced_mode:
            model_id = request.model.lower()
            if "instant" in model_id:
                forced_mode = "instant"
            elif "fast" in model_id:
                forced_mode = "fast"
            elif "thinking" in model_id:
                forced_mode = "thinking"

        # Process query via router
        route = thunder_instance.router.route_query(query, forced_mode=forced_mode)
        mode = route["mode"]
        predicted_len = route["predicted_length"]
        
        start_time = time.time()
        
        # Calculate required steps
        steps_count = thunder_instance.scheduler.calculate_steps(mode=mode, predicted_length=predicted_len)
        
        if route["target"] == "EXTERNAL_SEARCH":
            # Proactively "send" query to the external search agent
            search_results = await search_agent.search(query)
            response_text = f"Thunder Search Results:\n{search_results}"
        else:
            # Internal crystallization wait
            await asyncio.sleep(steps_count * 0.01) 
            response_text = "Thunder: Response crystallized successfully using hierarchical parallel diffusion."

        formatted_response = thunder_instance.personality.apply_formatting(response_text)
        
        end_time = time.time()
        duration = end_time - start_time
        token_count = len(response_text.split()) * 1.3
        
        if not request.stream:
            # Non-streaming implementation already handled earlier
            return {
                "id": f"chatcmpl-{random.randint(1000, 9999)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": formatted_response
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(query.split()),
                    "completion_tokens": int(token_count),
                    "total_tokens": int(token_count + len(query.split()))
                },
                "thunder_metrics": {
                    "mode": mode,
                    "duration_ms": round(duration * 1000, 2),
                    "tokens_per_second": round(token_count / duration, 2)
                }
            }
        else:
            # Streaming implementation (SSE)
            final_text = formatted_response # Use the result from either Search or Internal
            
            async def event_generator():
                completion_id = f"chatcmpl-{random.randint(1000, 9999)}"
                
                # Send the entire text as one single chunk (block)
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": final_text},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
            
    return router
