import re
from typing import List, Dict, Any

def split_passages(
    passage_id: str,
    passage_text: str,
    language: str = "kn",
    target_tokens: int = 200,
    max_tokens: int = 256,
    overlap_tokens: int = 40
) -> List[Dict[str, Any]]:
    """
    STRATEGY 2: Hybrid Fixed-Size + Overlap Chunking
    
    Splits passage text into overlapping chunks on sentence boundaries.
    Never cuts mid-sentence. Extends to nearest boundary (. ? ! । 。 ？ ！).
    If passage < 200 tokens, returns as a single chunk.
    """
    words = passage_text.strip().split()
    total_tokens = len(words)

    if total_tokens <= target_tokens:
        return [{
            "chunk_id": f"{passage_id}_c0",
            "text": passage_text.strip(),
            "overlap_with_prev": False,
            "token_count": total_tokens,
            "metadata": {
                "passage_id": passage_id,
                "language": language,
                "chunk_index": 0,
                "total_chunks": 1
            }
        }]

    # Sentence boundary regex covering English, Hindi/Devanagari, Telugu, CJK
    sentence_end_regex = re.compile(r'([.?!।|。？！]+)')

    raw_splits = sentence_end_regex.split(passage_text)
    sentences = []
    for i in range(0, len(raw_splits) - 1, 2):
        sentences.append(raw_splits[i] + raw_splits[i+1])
    if len(raw_splits) % 2 == 1 and raw_splits[-1].strip():
        sentences.append(raw_splits[-1])

    if not sentences:
        sentences = [passage_text]

    chunks = []
    current_words: List[str] = []
    chunk_index = 0

    for sentence in sentences:
        s_words = sentence.strip().split()
        if not s_words:
            continue

        if len(current_words) + len(s_words) > max_tokens and current_words:
            chunk_text = " ".join(current_words)
            chunks.append({
                "chunk_id": f"{passage_id}_c{chunk_index}",
                "text": chunk_text,
                "overlap_with_prev": chunk_index > 0,
                "token_count": len(current_words),
                "metadata": {
                    "passage_id": passage_id,
                    "language": language,
                    "chunk_index": chunk_index,
                    "total_chunks": 0
                }
            })

            # Retain overlap_tokens from end of current chunk
            overlap = current_words[-overlap_tokens:] if len(current_words) >= overlap_tokens else current_words[:]
            current_words = overlap + s_words
            chunk_index += 1
        else:
            current_words.extend(s_words)

    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append({
            "chunk_id": f"{passage_id}_c{chunk_index}",
            "text": chunk_text,
            "overlap_with_prev": chunk_index > 0,
            "token_count": len(current_words),
            "metadata": {
                "passage_id": passage_id,
                "language": language,
                "chunk_index": chunk_index,
                "total_chunks": len(chunks) + 1
            }
        })

    total_chunks_count = len(chunks)
    for c in chunks:
        c["metadata"]["total_chunks"] = total_chunks_count

    return chunks
