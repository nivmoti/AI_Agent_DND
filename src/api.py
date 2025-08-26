"""FastAPI server exposing the RAG DnD agent.
Run: uvicorn src.api:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .agent import RagAgent

app = FastAPI(title="AI DnD RAG Agent", version="0.1.0")

# Simple session -> agent cache
_agents = {}

def get_agent(session: str) -> RagAgent:
    ag = _agents.get(session)
    if ag is None:
        ag = RagAgent(session=session)
        _agents[session] = ag
    return ag

class ChatRequest(BaseModel):
    session: str = "default"
    message: str

class ChatResponse(BaseModel):
    session: str
    answer: str

class RememberRequest(BaseModel):
    session: str = "default"
    fact: str

class MemoryResponse(BaseModel):
    session: str
    facts: List[str]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    agent = get_agent(req.session)
    try:
        answer = agent.answer(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(session=req.session, answer=answer)

@app.post("/remember", response_model=MemoryResponse)
async def remember(req: RememberRequest):
    agent = get_agent(req.session)
    fact_text = req.fact.strip()
    if not fact_text:
        raise HTTPException(status_code=400, detail="Empty fact")
    # Reuse the memory add mechanism directly
    agent.memory.add_fact(fact_text)
    return MemoryResponse(session=req.session, facts=agent.memory.facts)

@app.get("/memory", response_model=MemoryResponse)
async def memory(session: str = "default"):
    agent = get_agent(session)
    return MemoryResponse(session=session, facts=agent.memory.facts)
