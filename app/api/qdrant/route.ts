import { NextResponse } from "next/server";
import { execFile } from "child_process";
import path from "path";

export interface QdrantSearchResult {
  id: string | number;
  score: number;
  payload?: Record<string, any>;
}

function searchLocalQdrant(
  query: string | number[],
  limit: number,
  collection: string
): Promise<QdrantSearchResult[]> {
  return new Promise((resolve) => {
    const scriptPath = path.join(process.cwd(), "app", "api", "store_data", "query_qdrant.py");
    const payload = typeof query === "string" ? { text: query, limit, collection } : { vector: query, limit, collection };

    const child = execFile(
      "python",
      [scriptPath],
      { maxBuffer: 10 * 1024 * 1024, timeout: 30000 },
      (error, stdout) => {
        if (error) {
          return resolve([]);
        }
        try {
          const results = JSON.parse(stdout.trim());
          if (Array.isArray(results)) {
            return resolve(results);
          }
        } catch (e) {}
        resolve([]);
      }
    );

    if (child.stdin) {
      child.stdin.write(JSON.stringify(payload));
      child.stdin.end();
    }
  });
}

export async function search(
  query: string | number[],
  limit: number = 5,
  collection?: string
): Promise<QdrantSearchResult[]> {
  const collectionName = collection || process.env.QDRANT_COLLECTION || "voice_rag";
  const results = await searchLocalQdrant(query, limit, collectionName);
  return results;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const queryInput = body.query || body.vector || body.text;
    const limit = body.limit || body.topK || 5;
    const collection = body.collection || body.collectionName;

    if (!queryInput) {
      return NextResponse.json(
        { error: "Field 'query', 'text', or 'vector' is required for Qdrant search." },
        { status: 400 }
      );
    }

    const results = await search(queryInput, limit, collection);
    return NextResponse.json({
      success: true,
      count: results.length,
      results,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to execute Qdrant vector search." },
      { status: 500 }
    );
  }
}
