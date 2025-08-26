# AI DnD RAG Agent

An AI agent that answers D&D 5e SRD questions using Retrieval-Augmented Generation (RAG) over `SRD_CC_v5.2.1.pdf`. It keeps conversational memory and can store user-provided facts to extend its knowledge.

## Features
- PDF ingestion & chunking
- Embedding & vector search
- OpenAI GPT model for answer synthesis with cited sources
- Persistent conversation memory per session
- User-injected long-term memory ("remember this")
- Simple CLI chat loop

## Project Layout
```
SRD_CC_v5.2.1.pdf        # Original SRD PDF (not committed publicly)
/ data/                  # Processed text & chunks
/ vectorstore/           # Chroma / FAISS persistent index
/ memory/                # Saved conversations & user memory JSON
/ src/
    ingest.py            # Build / update vector store
    agent.py             # RAG agent logic & memory manager
    chat.py              # CLI entrypoint
    config.py            # Settings & helpers
```

## Quick Start
1. Create & activate a virtual environment (Python 3.11+ recommended).
2. Install deps: `pip install -r requirements.txt`
3. Put `SRD_CC_v5.2.1.pdf` in project root (already present).
4. Export your OpenAI key (`OPENAI_API_KEY`).
5. Ingest the SRD: `python -m src.ingest`
6. Chat: `python -m src.chat --session mycampaign`

Inside chat you can:
- Ask rules questions.
- Tell the agent to remember facts: `Remember this: The party hired a guide named Lira.`
- See what it remembers: `What do you remember?` or `list memory`
- Exit with `exit` or `quit`.

## Notes
- The agent will only cite content from the SRD plus remembered user facts.
- Re-run `ingest.py` if you change chunking parameters or add new source files.

## Future Ideas
- Web UI (FastAPI + React)
- Tool use (dice rolling, stat calculations)
- Multi-turn planning / LangGraph

---
MIT License
