import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stderr = open(os.devnull, "w")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient

def main():
    input_data = ""
    try:
        if not sys.stdin.isatty():
            input_data = sys.stdin.read().strip()
        elif len(sys.argv) > 1:
            input_data = sys.argv[1]
    except Exception:
        pass

    if not input_data and len(sys.argv) > 1:
        input_data = sys.argv[1]

    if not input_data:
        print(json.dumps([]))
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
            print(json.dumps([]))
            return

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        qdrant_db_path = os.path.join(project_root, "qdrant_db")

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        client = None
        try:
            temp_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=2)
            existing_cols = [c.name for c in temp_client.get_collections().collections]
            if collection_name in existing_cols:
                client = temp_client
        except Exception:
            pass

        if client is None:
            client = QdrantClient(path=qdrant_db_path)

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

        print(json.dumps(output))
    except Exception as e:
        print(json.dumps([{"error": str(e)}]))

if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
