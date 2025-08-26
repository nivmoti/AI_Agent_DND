"""RAG agent with conversation + long-term memory."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .config import VECTOR_DIR, MEMORY_DIR, settings

MEMORY_DIR.mkdir(exist_ok=True, parents=True)

@dataclass
class Turn:
    role: str
    content: str
    time: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class MemoryStore:
    """Stores conversation history + user-declared permanent facts."""
    def __init__(self, session: str):
        self.session = session
        self.file = MEMORY_DIR / f"{session}_history.jsonl"
        self.facts_file = MEMORY_DIR / f"{session}_facts.json"
        self.turns: List[Turn] = []
        self.facts: List[str] = []
        self._load()

    def _load(self):
        if self.file.exists():
            for line in self.file.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                    self.turns.append(Turn(**obj))
                except Exception:
                    continue
        if self.facts_file.exists():
            try:
                self.facts = json.loads(self.facts_file.read_text(encoding="utf-8"))
            except Exception:
                self.facts = []

    def append(self, role: str, content: str):
        t = Turn(role=role, content=content)
        self.turns.append(t)
        with self.file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(t.__dict__, ensure_ascii=False) + "\n")

    def add_fact(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            self.facts_file.write_text(json.dumps(self.facts, ensure_ascii=False, indent=2), encoding="utf-8")

    def recent_messages(self, limit: int = 15) -> List[Turn]:
        return self.turns[-limit:]


class RagAgent:
    def __init__(self, session: str = "default"):
        self.session = session
        self.memory = MemoryStore(session)
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
        self.vs = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=self.embeddings, collection_name="srd")
        self.llm = ChatOpenAI(model=settings.model, temperature=settings.temperature, openai_api_key=settings.openai_api_key)

    def _detect_memory_command(self, text: str) -> str | None:
        lowered = text.lower()
        triggers = ["remember this:", "remember:", "store this:", "save this:"]
        for trig in triggers:
            if lowered.startswith(trig):
                return text[len(trig):].strip()
        return None

    def _is_recall_request(self, text: str) -> bool:
        lowered = text.lower().strip()
        return lowered in {"what do you remember?", "what do you remember", "list memory", "show memory", "list facts"}

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        docs = self.vs.similarity_search(query, k=settings.top_k)
        results = []
        for d in docs:
            meta = d.metadata or {}
            results.append({
                "content": d.page_content,
                "source": meta.get("source"),
                "chunk_id": meta.get("chunk_id"),
                "index": meta.get("index")
            })
        return results

    def build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> List[Any]:
        system = (
            "You are a helpful Dungeon Master rules assistant. Answer strictly using the provided SRD context and remembered user facts. "
            "If the answer is not in the context, say you don't have that rule in the SRD. Cite chunk ids you used in parentheses like (chunks: id1, id2). "
            "Incorporate relevant user memory facts only when they relate to the question."
        )
        fact_block = "\n".join(f"- {f}" for f in self.memory.facts) if self.memory.facts else "(No user facts stored)"
        ctx_block_lines = []
        for c in contexts:
            ctx_block_lines.append(f"[chunk {c['chunk_id']}] {c['content']}")
        ctx_block = "\n\n".join(ctx_block_lines) if ctx_block_lines else "(No retrieval results)"
        recent = self.memory.recent_messages()
        messages: List[Any] = [SystemMessage(content=system)]
        if recent:
            convo_text = []
            for t in recent:
                prefix = 'User' if t.role == 'user' else 'Assistant'
                convo_text.append(f"{prefix}: {t.content}")
            messages.append(SystemMessage(content="Recent conversation:\n" + "\n".join(convo_text)))
        messages.append(SystemMessage(content="User stored facts:\n" + fact_block))
        messages.append(SystemMessage(content="SRD context chunks:\n" + ctx_block))
        messages.append(HumanMessage(content=question))
        return messages

    def answer(self, question: str) -> str:
        fact_to_store = self._detect_memory_command(question)
        if fact_to_store:
            self.memory.add_fact(fact_to_store)
            self.memory.append("user", question)
            ack = f"I'll remember that: '{fact_to_store}'"
            self.memory.append("assistant", ack)
            return ack
        if self._is_recall_request(question):
            self.memory.append("user", question)
            if self.memory.facts:
                facts_md = "\n".join(f"- {f}" for f in self.memory.facts)
                ans = f"I currently remember these user-provided facts:\n{facts_md}"
            else:
                ans = "I don't have any stored user facts yet. Use 'Remember this: <fact>' to add one."
            self.memory.append("assistant", ans)
            return ans
        contexts = self.retrieve(question)
        messages = self.build_prompt(question, contexts)
        resp = self.llm.invoke(messages)
        answer = resp.content
        self.memory.append("user", question)
        self.memory.append("assistant", answer)
        return answer


__all__ = ["RagAgent"]
