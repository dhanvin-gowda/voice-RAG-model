import { NextResponse } from "next/server";

export interface GuardrailResult {
  allowed: boolean;
  harmDetected: boolean;
  originalText: string;
  sanitizedText: string;
  reason?: string;
  flaggedCategories?: string[];
}

export async function guardrail(input: string): Promise<GuardrailResult> {
  if (!input || typeof input !== "string" || !input.trim()) {
    return {
      allowed: false,
      harmDetected: false,
      originalText: input || "",
      sanitizedText: "No response text available.",
      reason: "Input data is empty or invalid.",
      flaggedCategories: ["empty_input"],
    };
  }

  const trimmedText = input.trim();

  if (trimmedText.length > 10000) {
    return {
      allowed: false,
      harmDetected: true,
      originalText: trimmedText,
      sanitizedText: "[Response truncated due to length limits]",
      reason: "Input data exceeds maximum allowed length.",
      flaggedCategories: ["length_exceeded"],
    };
  }

  const promptInjectionRegex = /(ignore previous instructions|system prompt|disregard prior|bypass safety|jailbreak)/i;
  const isInjection = promptInjectionRegex.test(trimmedText);

  return {
    allowed: !isInjection,
    harmDetected: isInjection,
    originalText: trimmedText,
    sanitizedText: isInjection ? "[Harmful content removed: malicious prompt manipulation detected]" : trimmedText,
    reason: isInjection ? "Prompt manipulation detected." : "Input passed standard guardrail checks.",
    flaggedCategories: isInjection ? ["prompt_injection"] : [],
  };
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const inputData = body.text || body.data;

    if (!inputData) {
      return NextResponse.json(
        { error: "Field 'text' or 'data' is required for guardrail check." },
        { status: 400 }
      );
    }

    const result = await guardrail(inputData);
    return NextResponse.json(result);
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to process guardrail evaluation." },
      { status: 500 }
    );
  }
}
