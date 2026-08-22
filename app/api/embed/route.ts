import { NextResponse } from "next/server";

const EMBEDDING_MODELS = [
  "text-embedding-004",
  "gemini-embedding-001",
  "embedding-001",
];

export async function sendEmbedding(text: string): Promise<number[]>;
export async function sendEmbedding(text: string[]): Promise<number[][]>;
export async function sendEmbedding(text: string | string[]): Promise<number[] | number[][]> {
  const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "your_google_api_key_here") {
    throw new Error("GOOGLE_API_KEY (or GEMINI_API_KEY) is missing or not configured in .env");
  }

  let lastError: Error | null = null;

  for (const model of EMBEDDING_MODELS) {
    try {
      if (Array.isArray(text)) {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:batchEmbedContents?key=${apiKey}`;
        const requests = text.map((t) => ({
          model: `models/${model}`,
          content: { parts: [{ text: t }] },
        }));

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ requests }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Model ${model} Error (${response.status}): ${errorText}`);
        }

        const data = await response.json();
        return data.embeddings.map((e: { values: number[] }) => e.values);
      } else {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent?key=${apiKey}`;
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: `models/${model}`,
            content: { parts: [{ text }] },
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Model ${model} Error (${response.status}): ${errorText}`);
        }

        const data = await response.json();
        return data.embedding.values;
      }
    } catch (err: unknown) {
      lastError = err instanceof Error ? err : new Error(String(err));
    }
  }

  throw new Error(`All Google Embedding models failed. Last error: ${lastError?.message || String(lastError)}`);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    if (!text) {
      return NextResponse.json(
        { error: "Field 'text' is required for generating embeddings." },
        { status: 400 }
      );
    }

    const embedding = await sendEmbedding(text);
    return NextResponse.json({ embedding });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Failed to generate Google embedding.";
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}

