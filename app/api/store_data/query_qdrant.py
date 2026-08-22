import os
import sys
import json
import base64
import warnings

os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore")
try:
    sys.stderr = open(os.devnull, "w")
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

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

def main():
    input_data = ""
    if len(sys.argv) > 1 and sys.argv[1].strip():
        arg = sys.argv[1].strip()
        try:
            input_data = base64.b64decode(arg).decode("utf-8")
        except Exception:
            input_data = arg

    if not input_data and not sys.stdin.isatty():
        try:
            input_data = sys.stdin.read().strip()
        except Exception:
            pass

    if not input_data:
        print(json.dumps([], ensure_ascii=True))
        return

    try:
        try:
            parsed = json.loads(input_data)
        except Exception:
            parsed = input_data

        query_input = None
        vector_data = []
        top_k = 5
        collection_name = os.getenv("QDRANT_COLLECTION", "voice_rag")

        if isinstance(parsed, dict):
            query_input = parsed.get("text") or parsed.get("query")
            vector_data = parsed.get("vector", [])
            top_k = int(parsed.get("limit", 5))
            collection_name = parsed.get("collection") or os.getenv("QDRANT_COLLECTION", "voice_rag")
        elif isinstance(parsed, str):
            query_input = parsed
        else:
            vector_data = parsed

        if not vector_data and query_input:
            from sentence_transformers import SentenceTransformer
            embed_model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
            model = SentenceTransformer(embed_model_name)
            vector_data = model.encode(str(query_input), normalize_embeddings=True).tolist()

        if not vector_data:
            print(json.dumps([], ensure_ascii=True))
            return

        qdrant_url = os.getenv("QDRANT_URL", "").rstrip("/")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is not set. Add it to .env to connect to Qdrant Cloud.")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=15)
        client.get_collections()

        points = []
        if hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=collection_name,
                query=vector_data,
                limit=top_k,
                with_payload=True
            )
            points = res.points if hasattr(res, "points") else res
        elif hasattr(client, "search"):
            points = client.search(
                collection_name=collection_name,
                query_vector=vector_data,
                limit=top_k,
                with_payload=True
            )

        output = []
        for point in points:
            output.append({
                "id": getattr(point, "id", 0),
                "score": getattr(point, "score", 0.0),
                "payload": getattr(point, "payload", {}) or {}
            })

        print(json.dumps(output, ensure_ascii=True))
    except Exception as e:
        print(json.dumps([{"error": str(e)}], ensure_ascii=True))

if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

