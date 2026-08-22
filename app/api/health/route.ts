import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

interface QdrantHealth {
  reachable: boolean;
  collections?: string[];
  pointCount?: number | null;
  error?: string;
}

export async function GET() {
  const collection = process.env.QDRANT_COLLECTION || "voice_rag_gemini";

  const env = {
    hasSarvamKey: Boolean(process.env.SARVAM_API),
    hasGoogleApiKey: Boolean(
      process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY
    ),
    hasQdrantUrl: Boolean(process.env.QDRANT_URL),
    hasQdrantApiKey: Boolean(process.env.QDRANT_API_KEY),
  };

  let qdrant: QdrantHealth = { reachable: false };
  try {
    const url = (process.env.QDRANT_URL || "").replace(/\/+$/, "");
    const apiKey = process.env.QDRANT_API_KEY;

    if (!url || !apiKey) {
      throw new Error("QDRANT_URL and/or QDRANT_API_KEY are not set.");
    }

    const headers = { "api-key": apiKey };

    const colsRes = await fetch(`${url}/collections`, {
      headers,
      signal: AbortSignal.timeout(10000),
    });
    if (!colsRes.ok) {
      throw new Error(`Qdrant /collections failed with status ${colsRes.status}`);
    }
    const colsData = (await colsRes.json()) as {
      result?: { collections?: { name: string }[] };
    };
    const collections = (colsData.result?.collections || []).map((c) => c.name);

    let pointCount: number | null = null;
    if (collections.includes(collection)) {
      const infoRes = await fetch(`${url}/collections/${collection}`, {
        headers,
        signal: AbortSignal.timeout(10000),
      });
      if (infoRes.ok) {
        const info = (await infoRes.json()) as {
          result?: { points_count?: number };
        };
        pointCount = info.result?.points_count ?? null;
      }
    }

    qdrant = { reachable: true, collections, pointCount };
  } catch (error: unknown) {
    qdrant = {
      reachable: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }

  const allEnvPresent = Object.values(env).every(Boolean);
  const status = allEnvPresent && qdrant.reachable ? "ok" : "degraded";

  return NextResponse.json({
    status,
    timestamp: new Date().toISOString(),
    env,
    collection,
    qdrant,
  });
}
