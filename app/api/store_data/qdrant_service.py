import os
import sys
import json
import time
import warnings
from http.server import HTTPServer, BaseHTTPRequestHandler

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

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Warm-up model and Qdrant client on startup
embed_model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(embed_model_name)

qdrant_url = os.getenv("QDRANT_URL", "").rstrip("/")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url:
    print("[QdrantService] FATAL: QDRANT_URL is not set. Add it to .env to connect to Qdrant Cloud.", flush=True)
    sys.exit(1)

qdrant_client = None
for attempt in range(2):
    try:
        temp_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=15)
        temp_client.get_collections()
        qdrant_client = temp_client
        print(f"[QdrantService] Connected to Qdrant Cloud at {qdrant_url}", flush=True)
        break
    except Exception as e:
        if attempt == 0:
            time.sleep(3)
        else:
            print(f"[QdrantService] FATAL: Qdrant Cloud unreachable at {qdrant_url}: {e}", flush=True)
            sys.exit(1)

class QdrantServiceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path in ["/search", "/query"]:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""

            try:
                parsed = json.loads(body) if body else {}
                query_input = parsed.get("text") or parsed.get("query")
                vector_data = parsed.get("vector", [])
                top_k = int(parsed.get("limit", 5))
                collection_name = parsed.get("collection") or os.getenv("QDRANT_COLLECTION", "voice_rag")

                if not vector_data and query_input:
                    vector_data = model.encode(str(query_input), normalize_embeddings=True).tolist()

                output = []
                if vector_data:
                    points = []
                    if hasattr(qdrant_client, "query_points"):
                        res = qdrant_client.query_points(
                            collection_name=collection_name,
                            query=vector_data,
                            limit=top_k,
                            with_payload=True
                        )
                        points = res.points if hasattr(res, "points") else res
                    elif hasattr(qdrant_client, "search"):
                        points = qdrant_client.search(
                            collection_name=collection_name,
                            query_vector=vector_data,
                            limit=top_k,
                            with_payload=True
                        )

                    for point in points:
                        output.append({
                            "id": getattr(point, "id", 0),
                            "score": getattr(point, "score", 0.0),
                            "payload": getattr(point, "payload", {}) or {}
                        })

                response_bytes = json.dumps(output, ensure_ascii=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                err_bytes = json.dumps([{"error": str(e)}], ensure_ascii=True).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=5005):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, QdrantServiceHandler)
    print(f"[QdrantService] Service running on http://127.0.0.1:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()

if __name__ == "__main__":
    port = 5005
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
