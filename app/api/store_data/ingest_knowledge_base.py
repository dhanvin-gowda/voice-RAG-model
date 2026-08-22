import os
import sys
import time
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def ingest_dataset(limit=5000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    parquet_path = os.path.join(project_root, "kantrain.parquet")

    if not os.path.exists(parquet_path):
        print(f"Error: Could not find {parquet_path}")
        return

    print("=" * 65)
    print(f"[Knowledge Base Ingestion] Extracting up to {limit} valid Q&A entries from {parquet_path}")
    print("=" * 65)

    start_time = time.time()
    pf = pq.ParquetFile(parquet_path)

    passages = []
    
    # Read table from row group
    table = pf.read_row_group(0, columns=['query', 'Answer', 'Eng_Query', 'Eng_Answer'])
    rows = table.to_pylist()

    for idx, row in enumerate(rows):
        if len(passages) >= limit:
            break

        eng_q = str(row.get('Eng_Query') or '').strip()
        eng_a = str(row.get('Eng_Answer') or '').strip()
        q = str(row.get('query') or '').strip()
        a = str(row.get('Answer') or '').strip()

        # Skip entries where no valid answer is available
        if eng_a == 'No Answer Present.' and a == 'No Answer Present.':
            continue

        best_query = eng_q if eng_q else q
        best_answer = eng_a if (eng_a and eng_a != 'No Answer Present.') else a

        if not best_query or not best_answer:
            continue

        passage_text = f"Question: {best_query}\nAnswer: {best_answer}"

        passages.append({
            "id": f"doc_{len(passages)+1}",
            "passage": passage_text,
            "query": best_query,
            "Eng_Query": eng_q,
            "Eng_Answer": eng_a,
            "Answer": a,
            "text": best_answer,
            "language": "en" if eng_q else "kn"
        })

    extract_time = time.time() - start_time
    print(f"[OK] Extracted {len(passages)} valid Q&A entries in {extract_time:.2f} seconds.")

    if not passages:
        print("No passages found.")
        return

    print(f"\n[Embedding] Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[Embedding] Batch encoding {len(passages)} passages...")
    texts_to_embed = [p["passage"] for p in passages]
    embeddings = model.encode(texts_to_embed, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

    points = []
    for idx, (p, emb) in enumerate(zip(passages, embeddings), start=1):
        point_id = idx
        payload = {
            "chunk_id": p["id"],
            "text": p["text"],
            "passage": p["passage"],
            "query": p["query"],
            "Eng_Query": p["Eng_Query"],
            "Eng_Answer": p["Eng_Answer"],
            "Answer": p["Answer"],
            "language": p["language"]
        }
        points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

    collection_name = "voice_rag"
    vector_dim = len(points[0].vector)

    # 1. Upsert into Server Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    try:
        server_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5)
        existing_cols = [c.name for c in server_client.get_collections().collections]
        if collection_name in existing_cols:
            col_info = server_client.get_collection(collection_name)
            current_dim = col_info.config.params.vectors.size if hasattr(col_info.config.params.vectors, 'size') else None
            if current_dim and current_dim != vector_dim:
                server_client.delete_collection(collection_name)
                existing_cols.remove(collection_name)

        if collection_name not in existing_cols:
            server_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )

        # Batch upsert points
        batch_size = 500
        for b_start in range(0, len(points), batch_size):
            server_client.upsert(collection_name=collection_name, points=points[b_start:b_start+batch_size])
        print(f"[OK] Upserted {len(points)} points into Server Qdrant ('{collection_name}')")
    except Exception as err:
        print(f"[Warning] Could not upsert to server Qdrant: {err}")

    # 2. Upsert into Local Disk Qdrant DB
    local_db_path = os.path.join(project_root, "qdrant_db")
    try:
        disk_client = QdrantClient(path=local_db_path)
        existing_cols = [c.name for c in disk_client.get_collections().collections]
        if collection_name in existing_cols:
            col_info = disk_client.get_collection(collection_name)
            current_dim = col_info.config.params.vectors.size if hasattr(col_info.config.params.vectors, 'size') else None
            if current_dim and current_dim != vector_dim:
                disk_client.delete_collection(collection_name)
                existing_cols.remove(collection_name)

        if collection_name not in existing_cols:
            disk_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
            )

        batch_size = 500
        for b_start in range(0, len(points), batch_size):
            disk_client.upsert(collection_name=collection_name, points=points[b_start:b_start+batch_size])
        print(f"[OK] Upserted {len(points)} points into Local Disk Qdrant DB at {local_db_path}")
    except Exception as err:
        print(f"[Warning] Could not upsert to local disk Qdrant: {err}")

    print("=" * 65)
    print(f"SUCCESS: Ingested {len(points)} Q&A knowledge points into Qdrant Vector DB!")
    print("=" * 65)

if __name__ == "__main__":
    limit_val = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    ingest_dataset(limit=limit_val)
