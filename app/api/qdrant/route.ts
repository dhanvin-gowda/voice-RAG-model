import { NextResponse } from "next/server";
import { sendEmbedding } from "../embed/route";

export interface QdrantSearchResult {
  id: string | number;
  score: number;
  payload?: Record<string, unknown>;
}

interface QdrantPoint {
  id: string | number;
  score?: number;
  payload?: Record<string, unknown> | null;
}

interface QdrantQueryResponse {
  result?: {
    points?: QdrantPoint[];
  };
  status?: string;
}

function getQdrantConfig(): { url: string; apiKey: string; collection: string } {
  const url = (process.env.QDRANT_URL || "").replace(/\/+$/, "");
  const apiKey = process.env.QDRANT_API_KEY || "";
  const collection = process.env.QDRANT_COLLECTION || "voice_rag_gemini";

  if (!url) {
    throw new Error("QDRANT_URL is not set in environment variables.");
  }
  if (!apiKey) {
    throw new Error("QDRANT_API_KEY is not set in environment variables.");
  }

  return { url, apiKey, collection };
}

async function searchQdrantCloud(
  queryVector: number[],
  limit: number,
  collection: string
): Promise<QdrantSearchResult[]> {
  const { url, apiKey } = getQdrantConfig();

  const res = await fetch(`${url}/collections/${collection}/points/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "api-key": apiKey,
    },
    body: JSON.stringify({
      query: queryVector,
      limit,
      with_payload: true,
    }),
    signal: AbortSignal.timeout(15000),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(
      `Qdrant search failed (${res.status}) on collection '${collection}': ${errText.slice(0, 300)}`
    );
  }

  const data = (await res.json()) as QdrantQueryResponse;
  const points = data.result?.points || [];

  return points.map((p) => ({
    id: p.id,
    score: typeof p.score === "number" ? p.score : 0,
    payload: p.payload || {},
  }));
}

export async function search(
  query: string | number[],
  limit: number = 5,
  collection?: string
): Promise<QdrantSearchResult[]> {
  const targetCollection =
    collection || process.env.QDRANT_COLLECTION || "voice_rag_gemini";

  let vector: number[];
  if (Array.isArray(query)) {
    vector = query;
  } else {
    const trimmed = query.trim();
    if (!trimmed) {
      return [];
    }
    vector = await sendEmbedding(trimmed, "RETRIEVAL_QUERY");
  }

  return searchQdrantCloud(vector, limit, targetCollection);
}

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));
    const queryInput: string | number[] | undefined = body.query || body.vector || body.text;
    const limit: number = body.limit || body.topK || 5;
    const collection: string | undefined = body.collection || body.collectionName;

    if (!queryInput) {
      return NextResponse.json(
        { error: "Field 'query', 'text', or 'vector' is required for Qdrant search." },
        { status: 400 }
      );
    }

    let config: { hasQdrantUrl: boolean; hasQdrantApiKey: boolean; collection: string };
    try {
      const cfg = getQdrantConfig();
      config = {
        hasQdrantUrl: true,
        hasQdrantApiKey: true,
        collection: collection || cfg.collection,
      };
    } catch (e: unknown) {
      return NextResponse.json(
        {
          success: false,
          error: e instanceof Error ? e.message : String(e),
          results: [],
          count: 0,
        },
        { status: 500 }
      );
    }

    try {
      const results = await search(queryInput, limit, collection);
      return NextResponse.json({
        success: true,
        count: results.length,
        results,
      });
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to execute Qdrant vector search.";
      console.error("[qdrant] Search failed:", errorMessage);
      return NextResponse.json(
        { success: false, error: errorMessage, ...config, results: [], count: 0 },
        { status: 502 }
      );
    }
  } catch (error: unknown) {
    const errorMessage =
      error instanceof Error ? error.message : "Failed to execute Qdrant vector search.";
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}
