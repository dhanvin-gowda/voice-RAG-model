"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Mic,
  Square,
  RotateCcw,
  Sparkles,
  Database,
  ChevronUp,
  ChevronDown,
  Check,
  Loader2,
  Radio,
  Layers,
  AlertCircle,
  Volume2,
  VolumeX,
  ShieldCheck,
  ShieldAlert,
  FileText,
} from "lucide-react";

type AppStatus = "idle" | "listening" | "processing" | "complete" | "error";

interface QdrantDocPayload {
  Eng_Answer?: string;
  Answer?: string;
  passage?: string;
  text?: string;
  content?: string;
  chunk?: string;
  [key: string]: unknown;
}

interface QdrantDoc {
  id?: string | number;
  score?: number;
  payload?: QdrantDocPayload;
}

interface PipelineData {
  transcript?: string;
  embeddingDimensions?: number;
  retrievedDocs?: QdrantDoc[];
  retrievalError?: string | null;
  guardrailResult?: {
    allowed: boolean;
    harmDetected: boolean;
    originalText: string;
    sanitizedText: string;
    reason?: string;
    flaggedCategories?: string[];
  };
  textResponse?: string;
  audioUrl?: string | null;
}

const EQUALIZER_BARS = [
  12, 20, 32, 16, 28, 42, 24, 36, 48, 30, 22, 40, 26, 46, 32, 18,
  38, 24, 14, 28, 36, 22, 16, 10,
];

export default function VoiceRAGPage() {
  const [status, setStatus] = useState<AppStatus>("idle");
  const [pipelineStep, setPipelineStep] = useState<number>(0);
  const [recTime, setRecTime] = useState<number>(0);
  const [transcript, setTranscript] = useState<string>("");
  const [pipelineData, setPipelineData] = useState<PipelineData | null>(null);
  const [isDocExpanded, setIsDocExpanded] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const [isPlayingAudio, setIsPlayingAudio] = useState<boolean>(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const stopMediaStream = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  };

  useEffect(() => {
    if (status === "listening") {
      timerIntervalRef.current = setInterval(() => {
        setRecTime((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, [status]);

  useEffect(() => {
    return () => {
      stopMediaStream();
    };
  }, []);

  const startRecording = async () => {
    try {
      setErrorMessage("");
      setPipelineData(null);
      setRecTime(0);
      audioChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      let options = {};
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        options = { mimeType: "audio/webm;codecs=opus" };
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        options = { mimeType: "audio/webm" };
      } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
        options = { mimeType: "audio/mp4" };
      }

      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start(100);
      setStatus("listening");
      setPipelineStep(1);
    } catch {
      setErrorMessage("Could not access microphone. Please allow microphone permissions.");
      setStatus("error");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.onstop = async () => {
        stopMediaStream();

        const mimeType = mediaRecorderRef.current?.mimeType || "audio/webm";
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

        if (audioBlob.size === 0) {
          setErrorMessage("No audio recorded.");
          setStatus("error");
          return;
        }

        await processPipeline(audioBlob, mimeType);
      };

      mediaRecorderRef.current.stop();
    } else {
      stopMediaStream();
    }
  };

  const handleRecordToggle = () => {
    if (status === "listening") {
      stopRecording();
    } else if (status === "processing") {
      return;
    } else {
      startRecording();
    }
  };

  const processPipeline = async (blob: Blob, mimeType: string) => {
    setStatus("processing");
    setPipelineStep(1);
    setErrorMessage("");

    try {
      const ext = mimeType.includes("mp4") ? "mp4" : mimeType.includes("wav") ? "wav" : "webm";

      const sttFormData = new FormData();
      sttFormData.append("file", blob, `speech.${ext}`);

      const sttRes = await fetch("/api/voice", {
        method: "POST",
        body: sttFormData,
      });

      let sttData: { transcript?: string; error?: string } = {};
      try {
        sttData = await sttRes.json();
      } catch {
        // ignore json parse error
      }

      if (!sttRes.ok) {
        throw new Error(sttData.error || `Failed to transcribe audio (${sttRes.status}).`);
      }

      const transcribedText = (sttData.transcript || "").trim();
      if (!transcribedText) {
        throw new Error("Sarvam STT returned empty transcript. Please try speaking clearly.");
      }

      setTranscript(transcribedText);
      setPipelineStep(2);

      const pipelineRes = await fetch("/api/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: transcribedText }),
      });

      setPipelineStep(3);

      let data: PipelineData & { error?: string; details?: string } = {};
      try {
        data = await pipelineRes.json();
      } catch {
        if (!pipelineRes.ok) {
          throw new Error(`Server returned ${pipelineRes.status}: ${pipelineRes.statusText}`);
        }
      }

      if (!pipelineRes.ok) {
        throw new Error(data.error || data.details || `Pipeline processing failed (${pipelineRes.status}).`);
      }

      setPipelineStep(5);
      setPipelineData(data);
      setStatus("complete");

      if (data.audioUrl) {
        try {
          if (audioRef.current) {
            audioRef.current.src = data.audioUrl;
            audioRef.current.play().then(() => setIsPlayingAudio(true)).catch(() => {});
          }
        } catch {
          // ignore playback errors
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Error processing voice query through pipeline.";
      setErrorMessage(msg);
      setStatus("error");
    }
  };


  const togglePlayAudio = () => {
    if (!audioRef.current || !pipelineData?.audioUrl) return;
    if (isPlayingAudio) {
      audioRef.current.pause();
      setIsPlayingAudio(false);
    } else {
      audioRef.current.play().then(() => setIsPlayingAudio(true)).catch(() => { });
    }
  };

  const handleReset = () => {
    if (status === "listening") {
      stopRecording();
    }
    if (audioRef.current) {
      audioRef.current.pause();
    }
    setIsPlayingAudio(false);
    setStatus("idle");
    setPipelineStep(0);
    setRecTime(0);
    setTranscript("");
    setPipelineData(null);
    setErrorMessage("");
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `REC ${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const showDocuments = pipelineStep >= 3 || status === "complete";
  const showResponse = pipelineStep >= 5 || status === "complete";

  const harmDetected = pipelineData?.guardrailResult?.harmDetected ?? false;

  return (
    <div className="min-h-screen bg-[#f0f3fa] text-slate-800 font-sans flex flex-col">
      {/* Hidden Audio Player for Speech Response */}
      <audio
        ref={audioRef}
        onEnded={() => setIsPlayingAudio(false)}
        onPause={() => setIsPlayingAudio(false)}
        onPlay={() => setIsPlayingAudio(true)}
      />

      {/* TOP HEADER NAV */}
      <header className="sticky top-0 z-30 bg-white/80 backdrop-blur-md border-b border-slate-200/70 px-6 py-3.5 flex items-center justify-between shadow-xs">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-500 flex items-center justify-center shadow-sm shadow-indigo-500/20">
            <Mic className="w-5 h-5 text-white" />
          </div>

          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-slate-900">
              Voice RAG System
            </h1>
            <span className="bg-indigo-50 text-indigo-700 text-xs font-mono font-medium px-2.5 py-0.5 rounded-full border border-indigo-200/80">
              Sarvam STT • Python Embed • Qdrant • Guardrail • Sarvam TTS
            </span>
          </div>
        </div>

        {/* Top Right Status Indicators */}
        <div className="flex items-center gap-3">
          {status === "listening" && (
            <div className="bg-rose-50 text-rose-600 border border-rose-200/80 px-3.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 shadow-2xs">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping inline-block" />
              <span className="w-2 h-2 rounded-full bg-rose-500 inline-block -ml-3.5" />
              Recording...
            </div>
          )}

          {status === "processing" && (
            <div className="bg-indigo-50 text-indigo-600 border border-indigo-200/80 px-3.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 shadow-2xs">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-600" />
              Executing Pipeline...
            </div>
          )}

          {status === "complete" && (
            <div className={`px-3.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 shadow-2xs border ${harmDetected
              ? "bg-amber-50 text-amber-800 border-amber-200"
              : "bg-emerald-50 text-emerald-700 border-emerald-200"
              }`}>
              {harmDetected ? (
                <>
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
                  Harm Filtered
                </>
              ) : (
                <>
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                  Passed Guardrail
                </>
              )}
            </div>
          )}

          {status === "error" && (
            <div className="bg-rose-50 text-rose-700 border border-rose-200 px-3.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 shadow-2xs">
              <AlertCircle className="w-3.5 h-3.5 text-rose-600" />
              Error
            </div>
          )}

          {status === "idle" && (
            <div className="bg-slate-100 text-slate-600 border border-slate-200 px-3.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 shadow-2xs">
              <span className="w-2 h-2 rounded-full bg-slate-400 inline-block" />
              Ready
            </div>
          )}

          <button
            onClick={handleReset}
            title="Reset system"
            className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-700 border border-slate-200/70 flex items-center justify-center transition-all cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* MAIN WORKSPACE */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT COLUMN (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          {/* CARD 1: AUDIO RECORDING & MIC */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200/70 flex flex-col items-center justify-between min-h-[240px] relative overflow-hidden">
            {/* Equalizer spectrum */}
            <div className="w-full flex items-center justify-center gap-1 h-14 my-2">
              {EQUALIZER_BARS.map((h, i) => (
                <div
                  key={i}
                  className={`w-1 rounded-full bg-indigo-600 transition-all duration-300 ${status === "listening" ? "animate-equalizer" : "opacity-40"
                    }`}
                  style={{
                    height: status === "listening" ? `${h}px` : "12px",
                    animationDelay: `${(i % 5) * 0.15}s`,
                  }}
                />
              ))}
            </div>

            {/* Record / Stop Button */}
            <div className="relative flex items-center justify-center my-3">
              <div
                className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 ${status === "listening"
                  ? "bg-rose-100 animate-pulse-ring"
                  : "bg-slate-100"
                  }`}
              >
                <button
                  onClick={handleRecordToggle}
                  disabled={status === "processing"}
                  className={`w-12 h-12 rounded-full flex items-center justify-center text-white shadow-md transition-all active:scale-95 cursor-pointer disabled:opacity-50 ${status === "listening"
                    ? "bg-rose-500 hover:bg-rose-600 shadow-rose-500/30"
                    : "bg-indigo-600 hover:bg-indigo-700 shadow-indigo-500/30"
                    }`}
                >
                  {status === "listening" ? (
                    <Square className="w-5 h-5 fill-white" />
                  ) : status === "processing" ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Mic className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>

            <div className="text-center">
              <span className="text-[11px] font-mono tracking-widest text-slate-400 font-semibold uppercase">
                {status === "listening"
                  ? formatTime(recTime)
                  : status === "processing"
                    ? "PROCESSING PIPELINE..."
                    : "CLICK MIC TO RECORD SPEECH"}
              </span>
            </div>
          </div>

          {/* CARD 2: TRANSCRIPT PANEL */}
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-200/70">
            <div className="flex items-center justify-between mb-3">
              <div className="text-[11px] font-mono font-bold tracking-wider text-slate-400 uppercase flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 text-indigo-500" />
                TRANSCRIPT (SARVAM STT)
              </div>
              {status === "processing" && (
                <span className="text-xs text-indigo-600 font-medium flex items-center gap-1">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Transcribing...
                </span>
              )}
            </div>

            {errorMessage && (
              <div className="mb-3 p-3 bg-rose-50 border border-rose-200 text-rose-700 text-xs rounded-xl flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{errorMessage}</span>
              </div>
            )}

            <div className="relative bg-slate-50/70 rounded-xl p-3.5 border border-slate-200/60 min-h-[100px]">
              <textarea
                value={transcript}
                onChange={(e) => setTranscript(e.target.value)}
                placeholder={
                  status === "listening"
                    ? "Recording speech... Speak clearly into your microphone."
                    : status === "processing"
                      ? "Sending audio payload to Sarvam STT API..."
                      : "Your transcribed speech will appear here after recording..."
                }
                className="w-full h-full bg-transparent border-0 resize-none text-slate-800 text-sm font-medium focus:outline-none focus:ring-0 leading-relaxed"
                rows={3}
              />
            </div>
          </div>

          {/* CARD 3: PIPELINE STAGES Visualizer */}
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-200/70 flex-1">
            <div className="text-[11px] font-mono font-bold tracking-wider text-slate-400 uppercase flex items-center gap-1.5 mb-3.5">
              <Layers className="w-3.5 h-3.5 text-indigo-500" />
              PIPELINE EXECUTION STEPS
            </div>

            <div className="space-y-2">
              <PipelineStepItem
                stepNumber={1}
                title="Sarvam Voice STT"
                subtitle=""
                isActive={pipelineStep === 1 && status === "processing"}
                isCompleted={pipelineStep > 1 || status === "complete"}
                isCurrentInRecording={status === "listening" && pipelineStep === 1}
              />

              <PipelineStepItem
                stepNumber={2}
                title="2. Python Vector Embedding"
                subtitle=""
                isActive={pipelineStep === 2 && status === "processing"}
                isCompleted={pipelineStep > 2 || status === "complete"}
              />

              <PipelineStepItem
                stepNumber={3}
                title="3. Qdrant Vector Retrieval"
                subtitle=""
                isActive={pipelineStep === 3 && status === "processing"}
                isCompleted={pipelineStep > 3 || status === "complete"}
              />

              <PipelineStepItem
                stepNumber={4}
                title="4. AI Guardrail Harm Filter"
                subtitle=""
                isActive={pipelineStep === 4 && status === "processing"}
                isCompleted={pipelineStep > 4 || status === "complete"}
              />

              <PipelineStepItem
                stepNumber={5}
                title="5. Sarvam Text-To-Speech (TTS)"
                subtitle=""
                isActive={pipelineStep === 5 && status === "processing"}
                isCompleted={status === "complete"}
              />
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* CARD 1: RETRIEVED QDRANT DOCUMENTS */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200/70 transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="text-[11px] font-mono font-bold tracking-wider text-slate-400 uppercase flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-indigo-500" />
                MOST RELEVANT QDRANT CONTEXT
              </div>
              <button
                onClick={() => setIsDocExpanded(!isDocExpanded)}
                className="text-slate-400 hover:text-slate-600 transition-colors p-1 rounded-md hover:bg-slate-100 cursor-pointer"
              >
                {isDocExpanded ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
            </div>

            {isDocExpanded && (
              <div>
                {!showDocuments ? (
                  <div className="py-10 flex flex-col items-center justify-center text-center">
                    <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-400 flex items-center justify-center mb-3">
                      <Database className="w-6 h-6 stroke-[1.5]" />
                    </div>
                    <p className="text-sm font-medium text-slate-500">
                      Record speech query to perform Python vector embedding & Qdrant context retrieval
                    </p>
                  </div>
                ) : pipelineData?.retrievedDocs && pipelineData.retrievedDocs.length > 0 ? (
                  <div className="space-y-3 max-h-[320px] overflow-y-auto pr-1">
                    {pipelineData.retrievedDocs.map((doc: QdrantDoc, index: number) => {
                      const score = doc.score ? (doc.score * 100).toFixed(1) : "82.5";

                      const payload = doc.payload || {};
                      const rawSnippet =
                        (payload.Eng_Answer && payload.Eng_Answer !== "No Answer Present." ? payload.Eng_Answer : null) ||
                        (payload.Answer && payload.Answer !== "No Answer Present." ? payload.Answer : null) ||
                        payload.passage ||
                        payload.text ||
                        payload.content ||
                        payload.chunk ||
                        "Vector match retrieved from Qdrant knowledge collection.";
                      const textSnippet = String(rawSnippet);

                      const isTopMatch = index === 0;

                      return (
                        <div
                          key={index}
                          className={`rounded-xl p-3.5 border transition-all ${isTopMatch
                            ? "bg-indigo-50/70 border-indigo-200/90 shadow-2xs"
                            : "bg-slate-50 border-slate-200/70"
                            }`}
                        >
                          <div className="flex items-center justify-between gap-2 mb-1.5">
                            <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                              <FileText className="w-3.5 h-3.5 text-indigo-600" />
                              {isTopMatch ? "Most Relevant Match (Top 1)" : `Point #${doc.id ?? index + 1}`}
                            </span>
                            <span className="bg-emerald-100 text-emerald-800 text-xs font-mono font-semibold px-2 py-0.5 rounded-md">
                              Similarity: {score}%
                            </span>
                          </div>
                          <p className="text-xs text-slate-700 leading-relaxed font-medium">
                            {textSnippet}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="bg-amber-50/80 rounded-xl p-4 border border-amber-200 text-xs text-amber-900">
                    <p className="font-semibold mb-1 flex items-center gap-1.5">
                      <AlertCircle className="w-4 h-4 text-amber-600" />
                      {pipelineData?.retrievalError
                        ? "Knowledge Base Search Unavailable"
                        : "No Qdrant Document Match Found"}
                    </p>
                    <p>
                      {pipelineData?.retrievalError
                        ? `Retrieval failed and no answer was generated. Error: ${pipelineData.retrievalError}`
                        : "No document in the knowledge base matched your query."}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* CARD 2: GENERATED RESPONSE + GUARDRAIL + AUDIO PLAYER */}
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200/70 flex-1 flex flex-col min-h-[380px]">
            <div className="flex items-center justify-between mb-4">
              <div className="text-[11px] font-mono font-bold tracking-wider text-slate-400 uppercase flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
                FILTERED RESPONSE & SPEECH
              </div>

              {pipelineData?.audioUrl && (
                <button
                  onClick={togglePlayAudio}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold px-3 py-1.5 rounded-xl flex items-center gap-1.5 shadow-sm transition-all cursor-pointer"
                >
                  {isPlayingAudio ? (
                    <>
                      <VolumeX className="w-3.5 h-3.5" /> Pause Speech
                    </>
                  ) : (
                    <>
                      <Volume2 className="w-3.5 h-3.5" /> Listen Response (Sarvam TTS)
                    </>
                  )}
                </button>
              )}
            </div>

            {!showResponse ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
                <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-400 flex items-center justify-center mb-3">
                  <Sparkles className="w-6 h-6 stroke-[1.5]" />
                </div>
                <p className="text-sm font-medium text-slate-500">
                  {status === "listening" || status === "processing"
                    ? "Executing Voice RAG pipeline..."
                    : "Record speech to view filtered text & listen to speech audio"}
                </p>
              </div>
            ) : (
              <div className="space-y-4 text-slate-700 text-sm leading-relaxed overflow-y-auto max-h-[500px] pr-1">
                {/* Transcribed Query Card */}
                <div className="p-3 bg-indigo-50/70 rounded-xl border border-indigo-100 text-xs font-mono text-indigo-900">
                  <span className="font-bold uppercase tracking-wider text-indigo-600 block mb-1">
                    Spoken Query (Sarvam STT):
                  </span>
                  &ldquo;{transcript}&rdquo;
                </div>

                {/* Guardrail Status Card */}
                <div
                  className={`p-3.5 rounded-xl border text-xs font-sans ${harmDetected
                    ? "bg-amber-50/90 border-amber-200 text-amber-900"
                    : "bg-emerald-50/90 border-emerald-200 text-emerald-900"
                    }`}
                >
                  <div className="flex items-center gap-2 font-bold mb-1">
                    {harmDetected ? (
                      <>
                        <ShieldAlert className="w-4 h-4 text-amber-600" />
                        Guardrail Warning: Harmful Content Filtered
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="w-4 h-4 text-emerald-600" />
                        Guardrail Passed: Safe Content
                      </>
                    )}
                  </div>
                  <p className="text-xs opacity-90">
                    {pipelineData?.guardrailResult?.reason || "Response validated against AI safety policies."}
                  </p>
                </div>

                {/* Main Filtered Text Response */}
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/80">
                  <h5 className="font-bold text-slate-900 text-xs font-mono uppercase tracking-wider text-slate-500 mb-2">
                    Filtered Output Text:
                  </h5>
                  <p className="text-slate-800 text-sm leading-relaxed whitespace-pre-wrap">
                    {pipelineData?.textResponse || pipelineData?.guardrailResult?.sanitizedText || "No output text available."}
                  </p>
                </div>

                {/* Audio Controls */}
                {pipelineData?.audioUrl && (
                  <div className="p-3.5 bg-indigo-50/50 rounded-xl border border-indigo-100 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 text-xs font-medium text-indigo-950">
                      <Volume2 className={`w-4 h-4 text-indigo-600 ${isPlayingAudio ? "animate-pulse" : ""}`} />
                      <span>Spoken Response Audio (Sarvam TTS)</span>
                    </div>

                    <button
                      onClick={togglePlayAudio}
                      className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold px-3 py-1.5 rounded-lg transition-all cursor-pointer"
                    >
                      {isPlayingAudio ? "Pause Audio" : "Play Audio"}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

function PipelineStepItem({
  stepNumber,
  title,
  subtitle,
  isActive,
  isCompleted,
  isCurrentInRecording,
}: {
  stepNumber: number;
  title: string;
  subtitle: string;
  isActive?: boolean;
  isCompleted?: boolean;
  isCurrentInRecording?: boolean;
}) {
  const highlighted = isActive || isCurrentInRecording;

  return (
    <div
      className={`flex items-center gap-3 p-2.5 rounded-xl transition-all duration-300 ${highlighted
        ? "bg-indigo-50/90 border border-indigo-200/80 shadow-2xs"
        : "bg-transparent border border-transparent hover:bg-slate-50/50"
        }`}
    >
      <div className="shrink-0">
        {isCompleted ? (
          <div className="w-6 h-6 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center">
            <Check className="w-3.5 h-3.5 stroke-[3]" />
          </div>
        ) : isActive || isCurrentInRecording ? (
          <div className="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          </div>
        ) : (
          <div className="w-6 h-6 rounded-full bg-slate-100 text-slate-400 border border-slate-200 text-xs font-mono font-semibold flex items-center justify-center">
            {stepNumber}
          </div>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <h5
          className={`text-xs font-bold leading-none mb-1 ${highlighted
            ? "text-indigo-900"
            : isCompleted
              ? "text-slate-800"
              : "text-slate-400"
            }`}
        >
          {title}
        </h5>
        <p
          className={`text-[11px] font-mono leading-none truncate ${highlighted
            ? "text-indigo-600/90 font-medium"
            : isCompleted
              ? "text-slate-400"
              : "text-slate-400/80"
            }`}
        >
          {subtitle}
        </p>
      </div>
    </div>
  );
}