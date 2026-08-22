import os
import sys
import uuid

# Ensure local store_data folder is in Python import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indexer import build_index
from retriever import vector_search
from datasets import load_dataset

# Force UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
NUM_ROWS = 5
COLLECTION_NAME = "voice_rag"
DEFAULT_QDRANT_URL = "http://localhost:6333"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env_path = os.path.join(project_root, ".env")

if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

QDRANT_URL = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

print("=" * 65, flush=True)
print("MSMARCO-XI (5 Rows) -> 3-Strategy RAG -> Qdrant Pipeline", flush=True)
print("=" * 65, flush=True)

# 1. Stream 5 rows from HuggingFace MSMARCO-XI (train/kantrain.parquet)
print(f"[1/3] Streaming {NUM_ROWS} rows from HuggingFace (ai4bharat/MSMARCO-XI)...", flush=True)
dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    data_files={"train": "train/kantrain.parquet"},
    split="train",
    streaming=True
)

passages = []
for i, row in enumerate(dataset):
    if len(passages) >= NUM_ROWS:
        break

    passage_text = (
        row.get("passage") or
        row.get("passage_text") or
        row.get("context") or
        row.get("text") or
        ""
    )
    query_text = (
        row.get("query") or
        row.get("question") or
        row.get("query_text") or
        ""
    )

    if passage_text and str(passage_text).strip():
        passages.append({
            "passage_id": f"msmarco_kn_{i+1}",
            "passage": str(passage_text).strip(),
            "query": str(query_text).strip(),
            "url": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/blob/main/train/kantrain.parquet",
            "language": "kn"
        })

print(f"[OK] Extracted {len(passages)} Kannada passages from HuggingFace.", flush=True)
for idx, p in enumerate(passages, 1):
    print(f"  Row #{idx}: '{p['passage'][:70]}...'", flush=True)

# 2. Run Strategy 1 (Metadata-Aware Indexing) & Strategy 2 (Hybrid Sentence Chunking + Qdrant Indexing)
print(f"\n[2/3] Indexing {len(passages)} passages into Qdrant...", flush=True)
index_result = build_index(
    passages=passages,
    collection_name=COLLECTION_NAME,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)

print(f"[OK] Indexed {index_result['processed_passages']} passages into {index_result['total_chunks_indexed']} chunks in Qdrant collection '{index_result['collection']}'.", flush=True)

# 3. Test Strategy 2 & Strategy 3 (Language Filtered ANN Search + Late Chunking Re-Rank)
test_query = passages[0]["query"] if passages[0]["query"] else "ಕರ್ನಾಟಕದ ಮಾಹಿತಿ"
print(f"\n[3/3] Testing Search & Re-Rank Query: '{test_query}'...", flush=True)

search_result = vector_search(
    query_text=test_query,
    query_language="kn",
    top_k=5,
    collection_name=COLLECTION_NAME,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)

print("\n" + "=" * 65, flush=True)
print(f"RAG PIPELINE RESULTS for: '{test_query}'", flush=True)
print(f"Candidates Retrieved: {search_result['candidates_retrieved']} | Requires Fallback: {search_result['requires_fallback']}", flush=True)
print("=" * 65, flush=True)

for i, chunk in enumerate(search_result["answer_chunks"], 1):
    print(f"\n[Rank {i}] Chunk ID: {chunk['chunk_id']}", flush=True)
    print(f"       Final Score  : {chunk['final_score']} (Rel: {chunk['relevance']}, Ground: {chunk['grounding']}, Comp: {chunk['completeness']})", flush=True)
    print(f"       Text Snippet : '{chunk['text'][:100]}...'", flush=True)
    print(f"       Use for Answer: {chunk['use_for_answer']}", flush=True)

print("=" * 65, flush=True)
