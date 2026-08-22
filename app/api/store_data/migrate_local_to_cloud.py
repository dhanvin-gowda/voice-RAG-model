import os
import sys
import time

# Ensure local store_data folder is in Python import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

SCROLL_PAGE_SIZE = 256
UPSERT_BATCH_SIZE = 256

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
local_db_path = os.path.join(project_root, "qdrant_db")
env_path = os.path.join(project_root, ".env")

if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

QDRANT_URL = (os.getenv("QDRANT_URL") or "").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or ""
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "voice_rag")


def connect_cloud() -> QdrantClient:
    last_err = None
    for attempt in range(2):
        try:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=30)
            client.get_collections()
            print(f"[OK] Connected to Qdrant Cloud at {QDRANT_URL}")
            return client
        except Exception as e:
            last_err = e
            if attempt == 0 and "compatibility" in str(e).lower():
                try:
                    client = QdrantClient(
                        url=QDRANT_URL, api_key=QDRANT_API_KEY or None,
                        timeout=30, check_compatibility=False
                    )
                    client.get_collections()
                    print(f"[OK] Connected to Qdrant Cloud at {QDRANT_URL} (compatibility check disabled)")
                    return client
                except Exception as e2:
                    last_err = e2
            time.sleep(3)
    raise RuntimeError(f"Could not reach Qdrant Cloud at {QDRANT_URL}: {last_err}")


def main():
    print("=" * 65)
    print(f"Migrating local collection '{COLLECTION_NAME}' -> Qdrant Cloud")
    print("=" * 65)

    cloud = connect_cloud()

    try:
        local = QdrantClient(path=local_db_path)
    except Exception as e:
        raise RuntimeError(
            f"Cannot open local DB at {local_db_path} (another process may hold the lock): {e}"
        )

    local_cols = [c.name for c in local.get_collections().collections]
    if COLLECTION_NAME not in local_cols:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found in local DB at {local_db_path}")

    col_info = local.get_collection(COLLECTION_NAME)
    vector_size = col_info.config.params.vectors.size
    distance = col_info.config.params.vectors.distance
    local_count = local.count(COLLECTION_NAME, exact=True).count

    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
        "Manhattan": Distance.MANHATTAN,
    }
    cloud_distance = distance_map.get(str(distance).split(".")[-1], Distance.COSINE)

    print(f"[Local] Collection '{COLLECTION_NAME}': {local_count} points, dim={vector_size}, distance={cloud_distance}")

    existing_cols = [c.name for c in cloud.get_collections().collections]
    if COLLECTION_NAME in existing_cols:
        cloud_col = cloud.get_collection(COLLECTION_NAME)
        cloud_dim = cloud_col.config.params.vectors.size if hasattr(cloud_col.config.params.vectors, 'size') else None
        if cloud_dim and cloud_dim != vector_size:
            raise RuntimeError(
                f"Cloud collection '{COLLECTION_NAME}' has dim {cloud_dim} but local has {vector_size}. "
                f"Resolve the dimension mismatch manually before migrating."
            )
        mode = "merge/overwrite"
    else:
        cloud.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=cloud_distance)
        )
        print(f"[Cloud] Created collection '{COLLECTION_NAME}' (dim={vector_size}, distance={cloud_distance})")
        mode = "create"

    migrated_ids = []
    first_vector = None
    offset = None
    migrated_total = 0

    while True:
        points, next_offset = local.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        batch = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload or {})
            for p in points
        ]
        if first_vector is None:
            first_vector = list(points[0].vector)

        for b_start in range(0, len(batch), UPSERT_BATCH_SIZE):
            cloud.upsert(
                collection_name=COLLECTION_NAME,
                points=batch[b_start:b_start + UPSERT_BATCH_SIZE],
                wait=True
            )

        migrated_ids.extend(p.id for p in points)
        migrated_total += len(points)
        print(f"  [Progress] Migrated {migrated_total}/{local_count} points...", flush=True)

        if next_offset is None:
            break
        offset = next_offset

    cloud_count = cloud.count(COLLECTION_NAME, exact=True).count
    print(f"\n[Verify] Local points: {local_count} | Cloud points: {cloud_count}")

    sample_ok = False
    if first_vector:
        try:
            res = cloud.query_points(collection_name=COLLECTION_NAME, query=first_vector, limit=3)
            hits = res.points if hasattr(res, "points") else res
            sample_ok = len(hits) > 0 and str(hits[0].id) == str(migrated_ids[0])
            print(f"[Verify] Sample search top hit: id={hits[0].id if hits else None}, score={hits[0].score if hits else None}")
        except Exception as e:
            print(f"[Verify] Sample search failed: {e}")

    success = (cloud_count >= local_count) and (not first_vector or sample_ok)
    print("\n" + "=" * 65)
    if success:
        print(f"SUCCESS: Migration complete ({mode}). {migrated_total} points now on Qdrant Cloud.")
    else:
        print("WARNING: Migration finished with verification discrepancies. Review counts above.")
    print("=" * 65)

    local.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
