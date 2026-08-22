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
qdrant_db_path = os.path.join(project_root, "qdrant_db")
env_path = os.path.join(project_root, ".env")

# Load environment variables
google_api_key = ""
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() in ["GOOGLE_API_KEY", "GEMINI_API_KEY"]:
                    google_api_key = v.strip()

if not google_api_key:
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
    print(f"Ingesting '{train_file_path}' into Qdrant Local Vector DB...")
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

    # Clear old Qdrant disk cache to ensure 3072 dimension matching
    import shutil
    if os.path.exists(qdrant_db_path):
        try:
            shutil.rmtree(qdrant_db_path, ignore_errors=True)
            os.makedirs(qdrant_db_path, exist_ok=True)
        except Exception as e:
            print("Warning clearing db directory:", e)

    # Connect to local Qdrant database
    client = QdrantClient(path=qdrant_db_path)
    vector_dim = len(points[0].vector)

    for col in ["voice_rag", "voice_rag_model"]:
        client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        client.upsert(collection_name=col, points=points)
        print(f"[OK] Upserted {len(points)} points into Qdrant collection '{col}'.")

    print("\n" + "=" * 65)
    print("SUCCESS: train.txt data successfully stored into Qdrant Vector DB!")
    print("=" * 65)

if __name__ == "__main__":
    ingest()
