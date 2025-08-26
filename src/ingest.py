"""Ingest SRD PDF -> text -> vector store.
Run: python -m src.ingest
"""
from pathlib import Path
from typing import List
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from .config import DATA_DIR, VECTOR_DIR, settings, BASE_DIR
import hashlib
import json

SOURCE_PDF = BASE_DIR / "SRD_CC_v5.2.1.pdf"
RAW_TEXT_FILE = DATA_DIR / "srd_raw.txt"
CHUNKS_FILE = DATA_DIR / "srd_chunks.jsonl"


def load_pdf_text(pdf_path: Path) -> str:
    print(f"[INGEST] Extracting text from {pdf_path} ...")
    text = extract_text(str(pdf_path))
    print(f"[INGEST] Extracted {len(text)} characters")
    return text


def chunk_text(text: str) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n", " "]
    )
    docs = splitter.create_documents([text])
    chunks = []
    for i, d in enumerate(docs):
        content = d.page_content.strip()
        if not content:
            continue
        chunk_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        chunks.append({
            "id": chunk_id,
            "content": content,
            "meta": {"source": str(SOURCE_PDF.name), "index": i}
        })
    print(f"[INGEST] Produced {len(chunks)} chunks")
    return chunks


def persist_chunks(chunks: List[dict]):
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"[INGEST] Wrote chunks JSONL to {CHUNKS_FILE}")


def build_vectorstore(chunks: List[dict]):
    print("[INGEST] Creating embeddings & building Chroma store ...")
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
    texts = [c["content"] for c in chunks]
    metadatas = [c["meta"] | {"chunk_id": c["id"]} for c in chunks]
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
        metadatas=metadatas,
        collection_name="srd"
    )
    print(f"[INGEST] Vector store persisted at {VECTOR_DIR}")


def main():
    if not SOURCE_PDF.exists():
        raise SystemExit(f"Missing PDF at {SOURCE_PDF}")
    raw_text = load_pdf_text(SOURCE_PDF)
    RAW_TEXT_FILE.write_text(raw_text, encoding="utf-8")
    print(f"[INGEST] Saved raw text to {RAW_TEXT_FILE}")
    chunks = chunk_text(raw_text)
    persist_chunks(chunks)
    build_vectorstore(chunks)
    print("[INGEST] Done.")


if __name__ == "__main__":
    main()
