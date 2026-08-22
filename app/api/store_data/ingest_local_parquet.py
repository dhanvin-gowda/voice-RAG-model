import os
import sys
import time
import pyarrow.parquet as pq

# Ensure local store_data directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
from indexer import build_index

def ingest_local(limit=1000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    parquet_path = os.path.join(project_root, "kantrain.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"Error: Could not find {parquet_path}")
        return

    print("=" * 65)
    print(f"Reading {limit} rows from {parquet_path}")
    print("=" * 65)
    
    start_time = time.time()
    
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception as e:
        print(f"Failed to open parquet file: {e}")
        return

    passages = []
    
    # Read row groups until we hit the limit
    for i in range(pf.num_row_groups):
        if len(passages) >= limit:
            break
            
        try:
            # Read row group specifying non-nested columns to avoid PyArrow nested array conversion errors
            table = pf.read_row_group(i, columns=['query', 'Answer', 'Eng_Query', 'Eng_Answer'])
            rows = table.to_pylist()
        except Exception as e:
            print(f"Error reading row group {i}: {e}")
            continue

        for row in rows:
            if len(passages) >= limit:
                break
            
            passage_text = ""
            
            # Safely extract from nested passages structure
            if row.get('passages') and isinstance(row['passages'], dict):
                tp = row['passages'].get('Translated_passages')
                if tp and isinstance(tp, dict) and 'list' in tp:
                    list_items = tp['list']
                    if list_items and len(list_items) > 0 and 'element' in list_items[0]:
                        passage_text = list_items[0]['element']
            
            if not passage_text and row.get('Answer'):
                passage_text = row['Answer']
                
            if not passage_text and row.get('Eng_Answer'):
                passage_text = row['Eng_Answer']
                
            query_text = row.get('query') or row.get('Eng_Query') or ""
            
            if passage_text and str(passage_text).strip():
                passages.append({
                    "passage_id": f"msmarco_local_kn_{len(passages)+1}",
                    "passage": str(passage_text).strip(),
                    "query": str(query_text).strip(),
                    "url": "local_kantrain.parquet",
                    "language": "kn"
                })
                
    extract_time = time.time() - start_time
    print(f"[OK] Extracted {len(passages)} passages in {extract_time:.2f} seconds.")
    
    if len(passages) == 0:
        print("No valid passages found. Exiting.")
        return

    print(f"\n[Indexing] Embedding and storing {len(passages)} rows into Qdrant...")
    print("Executing Strategy 1 (Metadata-Aware) & Strategy 2 (Hybrid Sentence-Boundary Chunking)...")
    
    index_start = time.time()
    
    try:
        index_result = build_index(
            passages=passages,
            collection_name="voice_rag",
            qdrant_url="http://localhost:6333",
            qdrant_api_key=""
        )
        index_time = time.time() - index_start
        print(f"\n[OK] Indexed {index_result['processed_passages']} passages into {index_result['total_chunks_indexed']} chunks.")
        print("=" * 65)
        print("PERFORMANCE REPORT:")
        print(f"1. Data Extraction Time : {extract_time:.2f} seconds")
        print(f"2. Embedding + Qdrant   : {index_time:.2f} seconds")
        print(f"3. Total Execution Time : {(extract_time + index_time):.2f} seconds")
        print("=" * 65)

        # Strategy 3 Verification Test
        from retriever import vector_search
        test_query = passages[0]["query"] if passages[0].get("query") else "ಮಾಹಿತಿ"
        print(f"\n[Strategy 3 Test] Executing Vector Search & Late Chunking Re-Rank for: '{test_query}'")
        search_res = vector_search(
            query_text=test_query,
            query_language="kn",
            top_k=3,
            collection_name="voice_rag"
        )
        print(f"Candidates Retrieved: {search_res['candidates_retrieved']} | Requires Fallback: {search_res['requires_fallback']}")
        for rank, chunk in enumerate(search_res["answer_chunks"], 1):
            print(f"  [Rank {rank}] Score: {chunk['final_score']} | Snippet: {chunk['text'][:80]}...")
    except Exception as e:
        print(f"Error during indexing: {e}")

if __name__ == "__main__":
    ingest_local(limit=1000)
