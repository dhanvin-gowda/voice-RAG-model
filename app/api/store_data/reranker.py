import math
from typing import List, Dict, Any

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors in-process."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))

def calculate_completeness_score(text: str, token_count: int) -> float:
    """
    Evaluates how standalone the chunk text is without external context.
    Checks sentence capitalization, terminal punctuation, and word count.
    """
    if not text or not text.strip():
        return 0.0

    score = 0.5
    trimmed = text.strip()

    # Terminal punctuation bonus
    if trimmed[-1] in ".?!।|。？！":
        score += 0.25

    # Capitalization / Start of sentence bonus
    if trimmed[0].isupper() or not trimmed[0].isascii():
        score += 0.15

    # Token length ratio bonus (target ~150-200 words)
    if 80 <= token_count <= 250:
        score += 0.10

    return max(0.0, min(1.0, score))

def late_chunk_score(
    query_vector: List[float],
    candidate_chunks: List[Dict[str, Any]],
    score_threshold: float = 0.4,
    max_answer_chunks: int = 3
) -> Dict[str, Any]:
    """
    STRATEGY 3: Late Chunking Re-Rank
    
    Evaluates candidate chunks on:
      1. relevance (0.5 weight): Cosine similarity between query vector & chunk vector
      2. grounding (0.3 weight): Cosine similarity between chunk vector & full_passage_vector
      3. completeness (0.2 weight): Standalone readability score
      
    final_score = (relevance * 0.5) + (grounding * 0.3) + (completeness * 0.2)
    """
    scored_results = []

    for chunk in candidate_chunks:
        chunk_id = chunk.get("chunk_id") or chunk.get("id", "unknown")
        text = chunk.get("text", "")
        chunk_vector = chunk.get("vector") or []
        payload = chunk.get("payload") or {}
        full_passage_vector = payload.get("full_passage_vector") or chunk_vector

        # Dimension 1: Relevance (Query <-> Chunk Vector Similarity)
        relevance = chunk.get("score")
        if relevance is None:
            relevance = cosine_similarity(query_vector, chunk_vector)
        else:
            relevance = max(0.0, min(1.0, float(relevance)))

        # Dimension 2: Grounding (Chunk <-> Full Passage Vector Similarity)
        grounding = cosine_similarity(chunk_vector, full_passage_vector) if full_passage_vector else 0.8

        # Dimension 3: Completeness (Standalone structural completeness)
        token_count = payload.get("token_count", len(text.split()))
        completeness = calculate_completeness_score(text, token_count)

        # Final Score Formula: (rel * 0.5) + (ground * 0.3) + (comp * 0.2)
        final_score = (relevance * 0.5) + (grounding * 0.3) + (completeness * 0.2)
        final_score = round(final_score, 4)

        use_for_answer = final_score >= score_threshold

        scored_results.append({
            "chunk_id": chunk_id,
            "text": text,
            "final_score": final_score,
            "relevance": round(relevance, 4),
            "grounding": round(grounding, 4),
            "completeness": round(completeness, 4),
            "use_for_answer": use_for_answer,
            "payload": payload
        })

    # Sort candidates by final_score descending
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    # Apply Rules:
    # 1. Mark use_for_answer = False if final_score < 0.4
    # 2. Return max 3 chunks with use_for_answer = True
    # 3. Set requires_fallback = True if none score above 0.4
    answer_chunks = [c for c in scored_results if c["use_for_answer"]][:max_answer_chunks]
    requires_fallback = len(answer_chunks) == 0

    return {
        "requires_fallback": requires_fallback,
        "selected_answer_chunks": answer_chunks,
        "all_ranked_chunks": scored_results
    }
