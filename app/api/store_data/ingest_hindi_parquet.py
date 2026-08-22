import os
import sys
import time
import pyarrow.parquet as pq

# Ensure local store_data directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from indexer import build_index
from retriever import vector_search

LIMIT = 1000
COLLECTION_NAME = "voice_rag"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env_path = os.path.join(project_root, ".env")

if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

QDRANT_URL = os.getenv("QDRANT_URL", "").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

PARQUET_PATH = os.path.join(project_root, "hintrain.parquet")


def count_cloud_points() -> int:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=20)
    return client.count(COLLECTION_NAME, exact=True).count


def extract_passages(limit: int):
    pf = pq.ParquetFile(PARQUET_PATH)
    passages = []
    for batch in pf.iter_batches(batch_size=max(limit, 1000),
                                 columns=['query', 'Answer', 'Eng_Query', 'Eng_Answer']):
        rows = batch.to_pylist()
        for row in rows:
            if len(passages) >= limit:
                break

            passage_text = str(row.get('Answer') or '').strip()
            if not passage_text:
                passage_text = str(row.get('Eng_Answer') or '').strip()

            query_text = str(row.get('query') or '').strip()
            if not query_text:
                query_text = str(row.get('Eng_Query') or '').strip()

            if not passage_text:
                continue

            passages.append({
                "passage_id": f"msmarco_hi_{len(passages) + 1}",
                "passage": passage_text,
                "query": query_text,
                "url": "local_hintrain.parquet",
                "language": "hi"
            })
        if len(passages) >= limit:
            break
    return passages


def main():
    print("=" * 65)
    print(f"Hindi Ingestion: first {LIMIT} rows of hintrain.parquet -> Qdrant Cloud")
    print("=" * 65)

    if not os.path.exists(PARQUET_PATH):
        print(f"Error: Could not find {PARQUET_PATH}")
        sys.exit(1)

    try:
        points_before = count_cloud_points()
        print(f"[Cloud] '{COLLECTION_NAME}' points before: {points_before}")
    except Exception as e:
        print(f"[Cloud] Could not read pre-count ({e}); continuing...")
        points_before = None

    start_time = time.time()
    passages = extract_passages(LIMIT)
    extract_time = time.time() - start_time
    print(f"[OK] Extracted {len(passages)} Hindi passages in {extract_time:.2f} seconds.")
    if not passages:
        print("No valid passages found. Exiting.")
        sys.exit(1)

    preview = passages[0]['passage'][:70].replace('\n', ' ')
    print(f"  Sample row #1: query='{passages[0]['query'][:50]}' | passage='{preview}...'")

    print(f"\n[Indexing] Embedding & upserting into cloud collection '{COLLECTION_NAME}' (Strategy 1+2)...")
    index_start = time.time()
    index_result = build_index(
        passages=passages,
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY
    )
    index_time = time.time() - index_start
    chunks = index_result['total_chunks_indexed']
    print(f"\n[OK] Indexed {index_result['processed_passages']} passages into {chunks} chunks.")

    print("\n[Verify] Checking cloud state...")
    points_after = count_cloud_points()
    print(f"[Verify] Points before: {points_before} | after: {points_after} | delta: "
          f"{(points_after - points_before) if points_before is not None else 'n/a'} (expected ~{chunks})")

    test_query = next((p["query"] for p in passages if p["query"]), None)
    if test_query:
        print(f"\n[Strategy 3 Test] Vector search + re-rank for: '{test_query[:60]}'")
        search_res = vector_search(
            query_text=test_query,
            query_language="hi",
            top_k=3,
            collection_name=COLLECTION_NAME,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY
        )
        print(f"Candidates Retrieved: {search_res['candidates_retrieved']} | "
              f"Requires Fallback: {search_res['requires_fallback']}")
        for rank, chunk in enumerate(search_res["answer_chunks"], 1):
            snippet = chunk['text'][:80].replace('\n', ' ')
            print(f"  [Rank {rank}] Score: {chunk['final_score']} | Snippet: {snippet}...")

    print("=" * 65)
    print(f"DONE. Extraction {extract_time:.2f}s | Indexing {index_time:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
