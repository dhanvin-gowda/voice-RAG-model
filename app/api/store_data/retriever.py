import os
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure local directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from reranker import late_chunk_score

def vector_search(
    query_text: str,
    query_language: Optional[str] = None,
    top_k: int = 5,
    collection_name: str = "voice_rag",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    embed_model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    STRATEGY 2 (Query-Time Vector Search)
    + STRATEGY 3 (Late Chunking Re-Rank)
    """
    qdrant_url = (qdrant_url or os.getenv("QDRANT_URL", "")).rstrip("/")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

    print(f"[Retriever] Embedding query: '{query_text}' (Language: {query_language or 'all'})")
    model = SentenceTransformer(embed_model_name)
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    # Connect to Qdrant Cloud (cloud-only mode)
    if not qdrant_url:
        raise RuntimeError("QDRANT_URL is not set. Add it to .env to connect to Qdrant Cloud.")

    client = None
    last_err = None
    for attempt in range(2):
        try:
            temp_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=15)
            existing_cols = [c.name for c in temp_client.get_collections().collections]
            if collection_name not in existing_cols:
                raise ValueError(f"Collection '{collection_name}' not found on Qdrant Cloud")
            client = temp_client
            print(f"[Retriever] Connected to Qdrant Cloud at {qdrant_url}")
            break
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(3)

    if client is None:
        raise RuntimeError(f"Qdrant Cloud unreachable at {qdrant_url}: {last_err}")

    lang_filter = None
    if query_language and query_language != "all":
        lang_filter = Filter(
            must=[
                FieldCondition(
                    key="language",
                    match=MatchValue(value=query_language)
                )
            ]
        )

    # Search helper compatible with qdrant-client v1.19+ (query_points) and earlier (search)
    def perform_search(filter_obj=None):
        if hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=filter_obj,
                limit=top_k,
                with_payload=True,
                with_vectors=True
            )
            return res.points
        elif hasattr(client, "search"):
            return client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_obj,
                limit=top_k,
                with_payload=True,
                with_vectors=True
            )
        return []

    try:
        raw_search_results = perform_search(lang_filter)
        if not raw_search_results:
            raw_search_results = perform_search(None)
    except Exception as e:
        print(f"[Retriever] Language filter search fallback (No filter): {e}")
        raw_search_results = perform_search(None)

    # Format candidate top-5 chunks
    candidate_chunks = []
    for hit in raw_search_results:
        payload = hit.payload or {}
        candidate_chunks.append({
            "chunk_id": payload.get("chunk_id", str(hit.id)),
            "text": payload.get("text", ""),
            "score": hit.score,
            "vector": hit.vector,
            "payload": payload
        })

    # Execute Strategy 3: Late Chunking Re-Rank
    reranked_output = late_chunk_score(query_vector, candidate_chunks)

    return {
        "query": query_text,
        "language_filter": query_language,
        "candidates_retrieved": len(candidate_chunks),
        "requires_fallback": reranked_output["requires_fallback"],
        "answer_chunks": reranked_output["selected_answer_chunks"],
        "all_ranked_chunks": reranked_output["all_ranked_chunks"]
    }
