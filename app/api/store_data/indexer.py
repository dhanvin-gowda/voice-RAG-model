import os
import sys
import re
import time
import hashlib
from typing import List, Dict, Any, Optional

# Ensure local directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from chunker import split_passages

def infer_domain(url: str, content: str) -> str:
    """
    Infers domain from URL or passage content.
    Returns: 'wikipedia', 'news', 'medical', 'legal', or 'general'
    """
    url_lower = (url or "").lower()
    content_lower = (content or "").lower()

    if "wikipedia.org" in url_lower or "wiki" in url_lower:
        return "wikipedia"
    if any(k in url_lower or k in content_lower for k in ["medical", "health", "disease", "treatment", "ವೈದ್ಯಕೀಯ", "ಆರೋಗ್ಯ"]):
        return "medical"
    if any(k in url_lower or k in content_lower for k in ["law", "court", "legal", "statute", "ನ್ಯಾಯಾಲಯ", "ಕಾನೂನು"]):
        return "legal"
    if any(k in url_lower or k in content_lower for k in ["news", "times", "today", "express", "ಸುದ್ದಿ", "ವಾರ್ತೆ"]):
        return "news"
    return "general"

def process_passage_strategy_1(
    model: SentenceTransformer,
    passage_id: str,
    passage_text: str,
    language_code: str = "kn",
    source_url: str = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI"
) -> Dict[str, Any]:
    """
    STRATEGY 1: Metadata-Aware Passage Chunking
    Treats full passage as one block, prepends metadata header, generates full_passage_vector.
    """
    domain = infer_domain(source_url, passage_text)
    embedding_input = f"[lang: {language_code}] [url: {source_url}] [domain: {domain}] {passage_text.strip()}"

    # Embed prefixed input for semantic domain search
    meta_embedding = model.encode(embedding_input, normalize_embeddings=True).tolist()
    # Embed raw passage for Strategy 3 re-ranking grounding check
    raw_embedding = model.encode(passage_text.strip(), normalize_embeddings=True).tolist()

    return {
        "chunk_id": f"{passage_id}_p0",
        "embedding_input": embedding_input,
        "meta_vector": meta_embedding,
        "full_passage_vector": raw_embedding,
        "metadata": {
            "passage_id": passage_id,
            "language": language_code,
            "url": source_url,
            "domain": domain
        }
    }

def build_index(
    passages: List[Dict[str, Any]],
    collection_name: str = "voice_rag",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embed_model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Executes Strategy 1 (Metadata-Aware Passage Indexing) and Strategy 2 (Hybrid Chunking & Qdrant Upsert).
    """
    qdrant_url = (qdrant_url or os.getenv("QDRANT_URL", "")).rstrip("/")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    print(f"[Indexer] Loading SentenceTransformer model '{embed_model_name}'...")
    model = SentenceTransformer(embed_model_name)

    points: List[PointStruct] = []
    processed_count = 0
    total_chunks_indexed = 0

    for idx, p in enumerate(passages, start=1):
        passage_id = str(p.get("passage_id") or p.get("id") or f"passage_{idx}")
        passage_text = p.get("passage") or p.get("text") or ""
        language_code = p.get("language", "kn")
        source_url = p.get("url", "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI")

        if not passage_text.strip():
            continue

        # Strategy 1 Execution
        s1_out = process_passage_strategy_1(model, passage_id, passage_text, language_code, source_url)

        # Strategy 2 Execution (Sentence boundary fixed-size + overlap chunking)
        s2_chunks = split_passages(passage_id, passage_text, language_code)

        for chunk_idx, chunk in enumerate(s2_chunks):
            # Embed chunk text
            chunk_vector = model.encode(chunk["text"], normalize_embeddings=True).tolist()
            total_chunks_indexed += 1

            # Prepare Qdrant point struct
            point_id = int(hashlib.sha256(chunk["chunk_id"].encode("utf-8")).hexdigest(), 16) % (10**12)
            payload = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "language": language_code,
                "url": source_url,
                "domain": s1_out["metadata"]["domain"],
                "full_passage_vector": s1_out["full_passage_vector"],
                "overlap_with_prev": chunk["overlap_with_prev"],
                "token_count": chunk["token_count"],
                "metadata": chunk["metadata"]
            }

            points.append(PointStruct(id=point_id, vector=chunk_vector, payload=payload))

        processed_count += 1
        if processed_count % 50 == 0 or processed_count == len(passages):
            print(f"  [Indexer Progress] Processed & Embedded {processed_count}/{len(passages)} passages ({total_chunks_indexed} chunks)...", flush=True)

    # Connect to Qdrant Cloud (cloud-only mode)
    if not qdrant_url:
        raise RuntimeError("QDRANT_URL is not set. Add it to .env to connect to Qdrant Cloud.")

    client = None
    last_err = None
    for attempt in range(2):
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=60)
            client.get_collections()
            print(f"[Indexer] Connected to Qdrant Cloud at {qdrant_url}")
            break
        except Exception as e:
            last_err = e
            client = None
            if attempt == 0:
                time.sleep(3)

    if client is None:
        raise RuntimeError(f"Qdrant Cloud unreachable at {qdrant_url}: {last_err}")

    if points:
        vector_dim = len(points[0].vector)
        existing_cols = [c.name for c in client.get_collections().collections]
        if collection_name in existing_cols:
            col_info = client.get_collection(collection_name)
            current_dim = col_info.config.params.vectors.size if hasattr(col_info.config.params.vectors, 'size') else None
            if current_dim and current_dim != vector_dim:
                print(f"[Indexer] Collection '{collection_name}' has dimension {current_dim}, but new vector dimension is {vector_dim}. Recreating collection...")
                client.delete_collection(collection_name)
                existing_cols.remove(collection_name)

        if collection_name not in existing_cols:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )

        UPSERT_BATCH_SIZE = 256
        for b_start in range(0, len(points), UPSERT_BATCH_SIZE):
            client.upsert(
                collection_name=collection_name,
                points=points[b_start:b_start + UPSERT_BATCH_SIZE],
                wait=True
            )
            print(f"[Indexer] Upserted batch {b_start // UPSERT_BATCH_SIZE + 1} "
                  f"({min(b_start + UPSERT_BATCH_SIZE, len(points))}/{len(points)} points)", flush=True)

    return {
        "status": "success",
        "processed_passages": processed_count,
        "total_chunks_indexed": total_chunks_indexed,
        "collection": collection_name
    }
