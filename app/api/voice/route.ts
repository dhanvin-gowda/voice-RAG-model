import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const apiKey = process.env.SARVAM_API;
    if (!apiKey) {
      return NextResponse.json(
        { error: "SARVAM_API key is missing in environment variables (.env)." },
        { status: 500 }
      );
    }

    const formData = await request.formData();
    const audioFile = formData.get("file") as Blob | File | null;

    if (!audioFile) {
      return NextResponse.json(
        { error: "No audio file provided in request." },
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

    const sarvamFormData = new FormData();
    const blobPayload = new Blob([audioBuffer], { type: mimeType });
    sarvamFormData.append("file", blobPayload, fileName);
    sarvamFormData.append("model", "saaras:v3");
    sarvamFormData.append("mode", "transcribe");

    const response = await fetch("https://api.sarvam.ai/speech-to-text", {
      method: "POST",
      headers: {
        "api-subscription-key": apiKey,
      },
      body: sarvamFormData,
    });

    if (!response.ok) {
      const errorText = await response.text();

      const fallbackFormData = new FormData();
      fallbackFormData.append("file", new Blob([audioBuffer], { type: mimeType }), fileName);
      fallbackFormData.append("model", "saarika:v2.5");
      fallbackFormData.append("mode", "transcribe");

      const fallbackResponse = await fetch("https://api.sarvam.ai/speech-to-text", {
        method: "POST",
        headers: {
          "api-subscription-key": apiKey,
        },
        body: fallbackFormData,
      });

      if (fallbackResponse.ok) {
        const fallbackData = await fallbackResponse.json();
        return NextResponse.json({
          transcript: fallbackData.transcript || fallbackData.transcripted_text || "",
          language_code: fallbackData.language_code || "",
        });
      }

      let parsedErrorMessage = errorText;
      try {
        const parsed = JSON.parse(errorText);
        if (parsed?.error?.message) {
          parsedErrorMessage = parsed.error.message;
        }
      } catch {
        // ignore JSON parse error
      }

      return NextResponse.json(
        { error: parsedErrorMessage || `Sarvam STT API returned ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json({
      transcript: data.transcript || data.transcripted_text || "",
      language_code: data.language_code || "",
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Failed to process audio transcription.";
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}

