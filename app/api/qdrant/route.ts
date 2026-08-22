import { NextResponse } from "next/server";
import { execFile, spawn } from "child_process";
import path from "path";

export interface QdrantSearchResult {
  id: string | number;
  score: number;
  payload?: Record<string, unknown>;
}

let serviceStarting = false;

function ensureDaemonRunning() {
  if (serviceStarting) return;
  serviceStarting = true;
  const servicePath = path.join(process.cwd(), "app", "api", "store_data", "qdrant_service.py");
  try {
    const child = spawn("python", [servicePath], {
      detached: true,
      stdio: "ignore",
    });
    child.unref();
  } catch {
    serviceStarting = false;
  }
}

async function searchLocalQdrant(
  query: string | number[],
  limit: number,
  collection: string
): Promise<QdrantSearchResult[]> {
  const payload = typeof query === "string" ? { text: query, limit, collection } : { vector: query, limit, collection };

  // 1. Ultra-fast HTTP Daemon query (~30ms)
  try {
    const res = await fetch("http://127.0.0.1:5005/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        return data;
      }
    }
  } catch {
    // Service not running yet, auto-spawn background daemon for subsequent queries
    ensureDaemonRunning();
  }

  // 2. Cold-start Process Fallback
  return new Promise((resolve) => {
    const scriptPath = path.join(process.cwd(), "app", "api", "store_data", "query_qdrant.py");
    const b64Payload = Buffer.from(JSON.stringify(payload)).toString("base64");

    execFile(
      "python",
      [scriptPath, b64Payload],
      { maxBuffer: 10 * 1024 * 1024, timeout: 45000, encoding: "utf-8" },
      (error, stdout) => {
        if (error) {
          return resolve([]);
        }
        try {
          const results = JSON.parse(stdout.trim());
          if (Array.isArray(results)) {
            return resolve(results);
          }
        } catch {
          // ignore parse errors
        }
        resolve([]);
      }
    );
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
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Failed to execute Qdrant vector search.";
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}

