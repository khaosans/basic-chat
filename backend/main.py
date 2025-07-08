from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import logging
from contextlib import asynccontextmanager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.async_ollama import AsyncOllamaChat
from document_processor import DocumentProcessor
from config import DEFAULT_MODEL, SYSTEM_PROMPT
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
chat_instances = {}
doc_processor = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global doc_processor
    logger.info("🚀 Starting BasicChat API server...")
    doc_processor = DocumentProcessor()
    logger.info("✅ API server started successfully")
    yield
    logger.info("🛑 Shutting down API server...")
app = FastAPI(
    title="BasicChat API",
    description="Streaming chat API for BasicChat application",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = DEFAULT_MODEL
    reasoning_mode: Optional[str] = "Auto"
    session_id: Optional[str] = None
class ChatResponse(BaseModel):
    content: str
    session_id: str
    model: str
    reasoning_mode: str
@app.get("/")
async def root():
    return {"message": "BasicChat API is running! 🚀"}
@app.get("/health")
async def health_check():
    try:
        chat = AsyncOllamaChat(DEFAULT_MODEL)
        is_healthy = await chat.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "ollama_available": is_healthy,
            "model": DEFAULT_MODEL
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            message = request.get("message", "")
            model = request.get("model", DEFAULT_MODEL)
            reasoning_mode = request.get("reasoning_mode", "Auto")
            session_id = request.get("session_id", "default")
            if not message:
                await websocket.send_text(json.dumps({"error": "Message is required"}))
                continue
            if session_id not in chat_instances:
                chat_instances[session_id] = AsyncOllamaChat(model)
            await websocket.send_text(json.dumps({
                "type": "status",
                "message": "Processing...",
                "session_id": session_id
            }))
            try:
                async for chunk in chat_instances[session_id].query_stream({
                    "inputs": message,
                    "system": SYSTEM_PROMPT
                }):
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "content": chunk,
                        "session_id": session_id
                    }))
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "session_id": session_id,
                    "model": model,
                    "reasoning_mode": reasoning_mode
                }))
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e),
                    "session_id": session_id
                }))
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": str(e)
            }))
        except:
            pass
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if request.session_id not in chat_instances:
            chat_instances[request.session_id] = AsyncOllamaChat(request.model)
        response = await chat_instances[request.session_id].query({
            "inputs": request.message,
            "system": SYSTEM_PROMPT
        })
        return ChatResponse(
            content=response or "Sorry, I couldn't generate a response.",
            session_id=request.session_id,
            model=request.model,
            reasoning_mode=request.reasoning_mode
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/models")
async def get_models():
    try:
        from ollama_api import get_available_models
        models = get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {"models": [DEFAULT_MODEL]}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 