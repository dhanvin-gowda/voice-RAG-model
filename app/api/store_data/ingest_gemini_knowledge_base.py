import os
import sys
import time
import json
import base64
import urllib.request
import urllib.error
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

GEMINI_MODEL = "gemini-embedding-001"
GEMINI_OUTPUT_DIM = 768
GEMINI_BATCH_SIZE = 100
MAX_RETRIES = 8


def gemini_embed_batch(texts, api_key, task_type="RETRIEVAL_DOCUMENT"):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:batchEmbedContents?key={api_key}"
    )
    requests_payload = [
        {
            "model": f"models/{GEMINI_MODEL}",
            "content": {"parts": [{"text": t}]},
            "taskType": task_type,
            "outputDimensionality": GEMINI_OUTPUT_DIM,
        }
        for t in texts
    ]
    body = json.dumps({"requests": requests_payload}).encode("utf-8")

    last_error = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [e["values"] for e in data["embeddings"]]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {err_body[:300]}"
            if e.code in (429, 500, 503):
                wait = min(2 ** attempt * 5, 120)
                print(f"[WARN] Gemini API {e.code}, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini embedding failed: {last_error}")
        except Exception as e:
            last_error = str(e)
            wait = min(2 ** attempt * 2, 60)
            print(f"[WARN] Gemini API error ({last_error[:150]}), retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError(f"Gemini embedding failed after {MAX_RETRIES} retries: {last_error}")


def ingest_dataset(limit=5000):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if not api_key or api_key == "your_google_api_key_here":
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env.")

    parquet_path = os.path.join(project_root, "kantrain.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: Could not find {parquet_path}")
        return

    collection_name = os.getenv("QDRANT_COLLECTION_GEMINI", "voice_rag_gemini")

    print("=" * 65)
    print(f"[Gemini KB Ingestion] Extracting up to {limit} valid Q&A entries from kantrain.parquet")
    print(f"[Gemini KB Ingestion] Model: {GEMINI_MODEL}, Collection: '{collection_name}'")
    print("=" * 65)

    start_time = time.time()
    pf = pq.ParquetFile(parquet_path)
    table = pf.read_row_group(0, columns=['query', 'Answer', 'Eng_Query', 'Eng_Answer'])
    rows = table.to_pylist()

    passages = []
    seen_queries = set()
    for row in rows:
        if len(passages) >= limit:
            break

        eng_q = str(row.get('Eng_Query') or '').strip()
        eng_a = str(row.get('Eng_Answer') or '').strip()
        q = str(row.get('query') or '').strip()
        a = str(row.get('Answer') or '').strip()

        if eng_a == 'No Answer Present.' and a == 'No Answer Present.':
            continue

        best_query = eng_q if eng_q else q
        best_answer = eng_a if (eng_a and eng_a != 'No Answer Present.') else a

        if not best_query or not best_answer:
            continue

        dedupe_key = best_query.lower()
        if dedupe_key in seen_queries:
            continue
        seen_queries.add(dedupe_key)

        passages.append({
            "id": f"doc_{len(passages)+1}",
            "passage": f"Question: {best_query}\nAnswer: {best_answer}",
            "query": best_query,
            "Eng_Query": eng_q,
            "Eng_Answer": eng_a,
            "Answer": a,
            "text": best_answer,
            "language": "en" if eng_q else "kn",
        })

    extract_time = time.time() - start_time
    print(f"[OK] Extracted {len(passages)} valid Q&A entries in {extract_time:.2f} seconds.")

    if not passages:
        print("No passages found.")
        return

    qdrant_url = (os.getenv("QDRANT_URL") or "").rstrip("/")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    if not qdrant_url:
        raise RuntimeError("QDRANT_URL is not set. Add it to .env to connect to Qdrant Cloud.")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=30)
    client.get_collections()
    existing_cols = [c.name for c in client.get_collections().collections]
    if collection_name in existing_cols:
        col_info = client.get_collection(collection_name)
        current_dim = (
            col_info.config.params.vectors.size
            if hasattr(col_info.config.params.vectors, 'size')
            else None
        )
        if current_dim and current_dim != GEMINI_OUTPUT_DIM:
            print(f"[Index] Collection '{collection_name}' has dim {current_dim}, recreating with dim {GEMINI_OUTPUT_DIM}...")
            client.delete_collection(collection_name)
            existing_cols.remove(collection_name)

    if collection_name not in existing_cols:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=GEMINI_OUTPUT_DIM, distance=Distance.COSINE),
        )

    # Resume support: fetch chunk_ids already stored so an interrupted
    # run skips them instead of re-embedding everything.
    ingested_ids = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for rec in records:
            cid = (rec.payload or {}).get("chunk_id")
            if cid:
                ingested_ids.add(cid)
        if offset is None:
            break

    if ingested_ids:
        print(f"[Resume] Found {len(ingested_ids)} already-ingested points; skipping them.")

    pending = [p for p in passages if p["id"] not in ingested_ids]
    skipped = len(passages) - len(pending)
    if not pending:
        print("[OK] Nothing new to ingest. Knowledge base is already up to date.")
        return

    total_batches = (len(pending) + GEMINI_BATCH_SIZE - 1) // GEMINI_BATCH_SIZE
    ingest_start = time.time()
    ingested_now = 0

    for batch_idx, b_start in enumerate(range(0, len(pending), GEMINI_BATCH_SIZE), start=1):
        batch = pending[b_start:b_start + GEMINI_BATCH_SIZE]
        batch_vecs = gemini_embed_batch([p["passage"] for p in batch], api_key)

        # Upsert immediately so progress survives interruptions;
        # the resume check skips these on any rerun.
        points = []
        for p, emb in zip(batch, batch_vecs):
            numeric_id = int(str(p["id"]).split("_")[1])
            payload = {
                "chunk_id": p["id"],
                "text": p["text"],
                "passage": p["passage"],
                "query": p["query"],
                "Eng_Query": p["Eng_Query"],
                "Eng_Answer": p["Eng_Answer"],
                "Answer": p["Answer"],
                "language": p["language"],
                "embed_model": GEMINI_MODEL,
            }
            points.append(PointStruct(id=numeric_id, vector=emb, payload=payload))
        client.upsert(collection_name=collection_name, points=points)

        ingested_now += len(batch)
        elapsed = time.time() - ingest_start
        print(f"[Ingest] {ingested_now}/{len(pending)} upserted (batch {batch_idx}/{total_batches}, {elapsed:.0f}s elapsed)")
        if b_start + GEMINI_BATCH_SIZE < len(pending):
            time.sleep(2)

    count = client.count(collection_name=collection_name, exact=True).count
    print("=" * 65)
    print(f"SUCCESS: Ingested {ingested_now} new Q&A knowledge points into Qdrant Cloud ('{collection_name}', now {count} total, {skipped} skipped as already present).")
    print("=" * 65)


if __name__ == "__main__":
    limit_val = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    ingest_dataset(limit=limit_val)
