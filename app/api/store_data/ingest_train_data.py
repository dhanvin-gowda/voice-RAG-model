import os
import sys
import json
import urllib.request
from typing import List, Dict, Any

# Ensure local store_data folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Project Root and .env setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
train_file_path = os.path.join(project_root, "train", "train.txt")
env_path = os.path.join(project_root, ".env")

# Load environment variables
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

google_api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

def get_google_embedding(text: str) -> List[float]:
    """Generates embedding vector via Google Gemini API (gemini-embedding-001)"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={google_api_key}"
    payload = json.dumps({
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]}
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        return res_data["embedding"]["values"]

def ingest():
    print("=" * 65)
    print(f"Ingesting '{train_file_path}' into Qdrant Cloud...")
    print("=" * 65)

    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"File not found: {train_file_path}")

    with open(train_file_path, "r", encoding="utf-8") as f:
        raw_content = f.read().strip()

    print(f"Loaded raw content ({len(raw_content)} chars):")
    print(f"'{raw_content}'\n")

    # Split content into distinct facts/sentences
    # Standardize fact breaks
    cleaned = raw_content.replace(".Here", ". Here").replace(".Capital:", ".\nCapital:")
    cleaned = cleaned.replace(".States", ".\nStates").replace(".National", ".\nNational")
    cleaned = cleaned.replace(".Currency", ".\nCurrency").replace(".Father", ".\nFather")
    cleaned = cleaned.replace(".First", ".\nFirst").replace(".Independence", ".\nIndependence")
    cleaned = cleaned.replace(".Famous", ".\nFamous")

    raw_lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    facts = []
    for line in raw_lines:
        if ":" in line and not line.startswith("Here"):
            facts.append(line)
        else:
            # sentence split
            parts = [p.strip() for p in line.split(".") if p.strip()]
            for p in parts:
                if len(p) > 10:
                    facts.append(p + ".")

    # Remove duplicates
    facts = list(dict.fromkeys(facts))

    print(f"Extracted {len(facts)} facts/chunks to index:")
    for idx, fact in enumerate(facts, 1):
        print(f"  [{idx}] {fact}")

    # Generate Google Gemini Embeddings for each fact
    points: List[PointStruct] = []
    for idx, fact in enumerate(facts, 1):
        print(f"Generating Google Embedding for Chunk #{idx}...", end=" ", flush=True)
        vec = get_google_embedding(fact)
        print(f"Done (dim={len(vec)})")

        point_id = idx
        payload = {
            "chunk_id": f"india_fact_{idx}",
            "text": fact,
            "passage": fact,
            "source": "train.txt",
            "topic": "India Facts",
            "language": "en"
        }
        points.append(PointStruct(id=point_id, vector=vec, payload=payload))

    # Connect to Qdrant Cloud (cloud-only mode)
    qdrant_url = (os.getenv("QDRANT_URL") or "").rstrip("/")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

    if not qdrant_url:
        raise RuntimeError("QDRANT_URL is not set. Add it to .env to connect to Qdrant Cloud.")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=15)
    client.get_collections()

    vector_dim = len(points[0].vector)

    # These Gemini embeddings (3072-dim) live in their own collection to keep
    # the 384-dim 'voice_rag' RAG schema intact on the cloud.
    collection_name = "voice_rag_model"
    existing_cols = [c.name for c in client.get_collections().collections]
    if collection_name in existing_cols:
        col_info = client.get_collection(collection_name)
        current_dim = col_info.config.params.vectors.size if hasattr(col_info.config.params.vectors, 'size') else None
        if current_dim and current_dim != vector_dim:
            print(f"[Index] Collection '{collection_name}' has dim {current_dim}, recreating with dim {vector_dim}...")
            client.delete_collection(collection_name)
            existing_cols.remove(collection_name)

    if collection_name not in existing_cols:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"[OK] Upserted {len(points)} points into Qdrant Cloud collection '{collection_name}'.")

    print("\n" + "=" * 65)
    print("SUCCESS: train.txt data successfully stored into Qdrant Cloud!")
    print("=" * 65)

if __name__ == "__main__":
    ingest()
