import { NextResponse } from "next/server";
import { search as searchQdrant } from "../qdrant/route";
import { guardrail } from "../guardrail/route";

export async function POST(request: Request) {
  try {
    const sarvamApiKey = process.env.SARVAM_API;

    if (!sarvamApiKey) {
      return NextResponse.json(
        { error: "SARVAM_API key is missing in environment variables (.env)." },
        { status: 500 }
      );
    }

    let transcript = "";
    let isJson = false;
    let requestText = "";

    try {
      const contentType = request.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        isJson = true;
        const jsonBody = await request.json();
        requestText = (jsonBody.text || jsonBody.transcript || "").trim();
      }
    } catch (e) {}

    if (requestText) {
      transcript = requestText;
    } else {
      const formData = await request.formData();
      const directText = (formData.get("text") as string) || (formData.get("transcript") as string);
      if (directText && directText.trim()) {
        transcript = directText.trim();
      } else {
        const audioFile = formData.get("file") as Blob | File | null;

        if (!audioFile) {
          return NextResponse.json(
            { error: "No audio file or text provided in request." },
            { status: 400 }
          );
        }

        const fileName = (audioFile as File).name || "speech.webm";
        let mimeType = audioFile.type ? audioFile.type.split(";")[0].trim() : "audio/webm";
        if (!mimeType || mimeType === "application/octet-stream") {
          if (fileName.endsWith(".wav")) mimeType = "audio/wav";
          else if (fileName.endsWith(".mp4")) mimeType = "audio/mp4";
          else mimeType = "audio/webm";
        }

        const arrayBuffer = await audioFile.arrayBuffer();
        const audioBuffer = Buffer.from(arrayBuffer);

        let sarvamFormData = new FormData();
        sarvamFormData.append("file", new Blob([audioBuffer], { type: mimeType }), fileName);
        sarvamFormData.append("model", "saaras:v3");
        sarvamFormData.append("mode", "transcribe");

        let sttResponse = await fetch("https://api.sarvam.ai/speech-to-text", {
          method: "POST",
          headers: { "api-subscription-key": sarvamApiKey },
          body: sarvamFormData,
        });

        if (sttResponse.ok) {
          const sttData = await sttResponse.json();
          transcript = (sttData.transcript || sttData.transcripted_text || "").trim();
        } else {
          const fallbackFormData = new FormData();
          fallbackFormData.append("file", new Blob([audioBuffer], { type: mimeType }), fileName);
          fallbackFormData.append("model", "saarika:v2.5");
          fallbackFormData.append("mode", "transcribe");

          const fallbackResponse = await fetch("https://api.sarvam.ai/speech-to-text", {
            method: "POST",
            headers: { "api-subscription-key": sarvamApiKey },
            body: fallbackFormData,
          });

          if (fallbackResponse.ok) {
            const fallbackData = await fallbackResponse.json();
            transcript = (fallbackData.transcript || fallbackData.transcripted_text || "").trim();
          } else {
            const errText = await sttResponse.text();
            return NextResponse.json(
              { error: `Sarvam STT failed: ${errText}` },
              { status: sttResponse.status }
            );
          }
        }
      }
    }

    if (!transcript) {
      return NextResponse.json(
        { error: "Sarvam STT returned empty transcript. Please try speaking clearly." },
        { status: 400 }
      );
    }

    let qdrantDocs: any[] = [];
    try {
      const rawDocs = await searchQdrant(transcript, 5);
      if (Array.isArray(rawDocs)) {
        rawDocs.sort((a, b) => b.score - a.score);
        qdrantDocs = rawDocs;
      }
    } catch (qdrantErr: any) {}

    let rawAnswerText = "";

    if (qdrantDocs && qdrantDocs.length > 0) {
      for (const doc of qdrantDocs) {
        const payload = doc.payload || {};
        const candidates = [
          payload.Eng_Answer,
          payload.Answer,
          payload.text,
          payload.passage,
          payload.content,
        ];
        for (const cand of candidates) {
          if (
            cand &&
            typeof cand === "string" &&
            cand.trim() !== "" &&
            cand.trim() !== "No Answer Present."
          ) {
            rawAnswerText = cand.trim();
            break;
          }
        }
        if (rawAnswerText) break;
      }
    }

    if (!rawAnswerText || rawAnswerText.trim() === "") {
      rawAnswerText = `No document in the knowledge base matched your query: "${transcript}".`;
    }

    const guardrailResult = await guardrail(rawAnswerText);
    const finalFilteredText = guardrailResult.sanitizedText || rawAnswerText;

    let audioUrl: string | null = null;
    try {
      const ttsInputText = finalFilteredText.slice(0, 450);

      const ttsResponse = await fetch("https://api.sarvam.ai/text-to-speech", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "api-subscription-key": sarvamApiKey,
        },
        body: JSON.stringify({
          inputs: [ttsInputText],
          target_language_code: "en-IN",
          speaker: "anushka",
          pitch: 0,
          pace: 1.05,
          loudness: 1.5,
          speech_sample_rate: 8000,
          enable_preprocessing: true,
          model: "bulbul:v2",
        }),
      });

      if (ttsResponse.ok) {
        const ttsData = await ttsResponse.json();
        if (ttsData.audios && ttsData.audios.length > 0 && ttsData.audios[0]) {
          const base64Audio = ttsData.audios[0];
          audioUrl = base64Audio.startsWith("data:")
            ? base64Audio
            : `data:audio/wav;base64,${base64Audio}`;
        }
      }
    } catch (ttsErr) {}

    return NextResponse.json({
      success: true,
      transcript,
      retrievedDocs: qdrantDocs,
      guardrailResult,
      textResponse: finalFilteredText,
      audioUrl,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to execute Voice RAG pipeline." },
      { status: 500 }
    );
  }
}
