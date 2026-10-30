import os
import json
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

try:
    from pruner import TokenPruner
except ImportError:
    from backend.pruner import TokenPruner

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prompt_pruner")

app = FastAPI(
    title="Prompt Pruning & LLM Acceleration Studio",
    description="High-performance ONNX token pruning engine with real-time SSE streaming generation",
    version="1.0.0",
)

# Enable permissive CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the token pruning engine
pruner = TokenPruner()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class PruneRequest(BaseModel):
    text: str = Field(..., description="Prompt or context text to prune")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Retention probability cutoff")


class GenerateRequest(BaseModel):
    compressed_context: str = Field("", description="Pruned context string")
    query: str = Field(..., description="User query or instruction")
    model: str = Field("qwen2.5:7b", description="Ollama model name")


@app.get("/api/health")
async def health_check():
    """Health check endpoint reporting engine status and providers."""
    ollama_online = False
    available_models = []
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                ollama_online = True
                data = resp.json()
                available_models = [m.get("name") for m in data.get("models", [])]
    except Exception:
        pass

    return {
        "status": "online",
        "provider": pruner.active_provider,
        "ollama_online": ollama_online,
        "available_models": available_models,
    }


@app.get("/api/models")
async def list_models():
    """Fetch list of local models available in Ollama."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.warning(f"Could not reach Ollama at {OLLAMA_BASE_URL}: {e}")
    return {"models": []}


@app.post("/api/prune")
def prune_endpoint(payload: PruneRequest):
    """
    Evaluates token retention probabilities and returns structured token diffs
    and compression metrics.
    """
    try:
        result = pruner.prune(text=payload.text, threshold=payload.threshold)
        return result
    except Exception as e:
        logger.error(f"Prune error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate_endpoint(payload: GenerateRequest):
    """
    Forwards prompt composed of compressed_context and query to Ollama
    and streams tokens via Server-Sent Events (SSE).
    """
    # Verify local Ollama model availability or graceful fallback
    target_model = payload.model
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            tags_resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if tags_resp.status_code == 200:
                models_data = tags_resp.json().get("models", [])
                names = [m.get("name") for m in models_data]
                if target_model not in names:
                    # Match by base name if tag omitted
                    matching = [n for n in names if n.startswith(target_model.split(":")[0])]
                    if matching:
                        target_model = matching[0]
                    elif names:
                        logger.info(
                            f"Model '{payload.model}' not found in Ollama. Falling back to installed '{names[0]}'"
                        )
                        target_model = names[0]
    except Exception as e:
        logger.warning(f"Ollama tags lookup skipped: {e}")

    # Build prompt
    if payload.compressed_context.strip():
        full_prompt = (
            f"Context:\n{payload.compressed_context.strip()}\n\n"
            f"Question:\n{payload.query.strip()}\n\n"
            f"Answer:"
        )
    else:
        full_prompt = payload.query.strip()

    ollama_payload = {
        "model": target_model,
        "prompt": full_prompt,
        "stream": True,
    }

    async def sse_event_stream():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=ollama_payload,
                ) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        err_msg = json.dumps({"token": f"\n[Ollama Error {response.status_code}: {err_text.decode('utf-8')}]"})
                        yield f"data: {err_msg}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

            yield "data: [DONE]\n\n"
        except httpx.ConnectError:
            err_msg = json.dumps({
                "token": "\n[Error: Could not connect to Ollama at http://localhost:11434. Please ensure Ollama is running.]"
            })
            yield f"data: {err_msg}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as ex:
            logger.error(f"SSE stream error: {ex}", exc_info=True)
            err_msg = json.dumps({"token": f"\n[Stream Error: {str(ex)}]"})
            yield f"data: {err_msg}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
