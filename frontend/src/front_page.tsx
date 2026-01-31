import { useState } from "react";
import SlideAnimation from "./components/SlideAnimation";

type Probabilities = {
  negative: number;
  neutral: number;
  positive: number;
};

type ChunkSummary = {
  chunk_id: number;
  preview: string;
  predicted_label: string;
  probabilities: Probabilities;
};

type UploadResponse = {
  session_id: string | null;
  document: {
    predicted_label: string;
    probabilities: Probabilities;
    num_chunks: number;
  };
  chunks: ChunkSummary[];
  message?: string;
};

type ExplainResponse = {
  chunk_id: number;
  predicted_label: string;
  probabilities: Probabilities;
  top_word_contributions: [string, number][];
  error?: string;
};

function DisplayFrontPage() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [status, setStatus] = useState<string>("");

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setStatus(`Uploading: ${file.name}...`);
    setExplain(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`http://127.0.0.1:8000/upload-text?top_n=20`, {
        method: "POST",
        body: formData,
      });

      const bodyText = await res.text();
      if (!res.ok) {
        setStatus(`Error: (${res.status}): ${bodyText}`);
        return;
      }

      const data = JSON.parse(bodyText) as UploadResponse;
      setUpload(data);
      setStatus("");
    } catch (err) {
      setStatus(`Request failed: ${String(err)}`);
    } finally {
      e.target.value = "";
    }
  }

  async function handleExplain(chunkId: number) {
    if (!upload?.session_id) return;
    setStatus("Generating explanation...");
    setExplain(null);

    const res = await fetch(`http://127.0.0.1:8000/explain-chunk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: upload.session_id,
        chunk_id: chunkId,
        top_contribution_words: 15,
      }),
    });

    const data = (await res.json()) as ExplainResponse;
    setExplain(data);
    setStatus("");
  }

  return (
    <div className="min-h-screen p-6 bg-white text-gray-900">
      <div className="text-center py-8">
        <h1 className="text-3xl font-bold">Explain My Sentiment</h1>

        <div className="mt-4">
          <label className="cursor-pointer inline-flex items-center gap-3 px-6 py-3 rounded-lg border hover:bg-gray-50
              active:scale-[0.98] active:bg-gray-100 transition focus-within:outline-none focus-within:ring-2 focus-within:ring-gray-300 focus-within:ring-offset-2">
            <span>Upload text file</span>
            <input type="file" accept=".txt" className="hidden" onChange={handleFileUpload}/>
          </label>
        </div>

        {status && <div className="mt-3 text-sm opacity-70">{status}</div>}
      </div>

      {upload?.session_id && (
        <div className="max-w-5xl mx-auto">
          <SlideAnimation>
            <div className="border rounded-lg p-4">
              <div className="font-semibold">Document summary</div>
              <div className="mt-2 text-sm">
                Predicted label: 
                {upload.document.predicted_label === "positive" && (
                  <span className="font-mono px-2 py-0.5 rounded bg-green-100 text-green-800">Positive</span>
                )}
                {upload.document.predicted_label === "neutral" && (
                  <span className="font-mono px-2 py-0.5 rounded bg-gray-100 text-gray-800">Neutral</span>
                )}
                {upload.document.predicted_label === "negative" && (
                  <span className="font-mono px-2 py-0.5 rounded bg-red-100 text-red-800">Negative</span>
                )}
              </div>
              <div className="mt-1 text-sm font-mono">
                neg={upload.document.probabilities.negative.toFixed(3)}{" "}
                neu={upload.document.probabilities.neutral.toFixed(3)}{" "}
                pos={upload.document.probabilities.positive.toFixed(3)}
              </div>
            </div>
          </SlideAnimation>

          <SlideAnimation>     
            <div className="mt-6 border rounded-lg p-4">
              <div className="font-semibold">
                Choose a chunk to explain (Top {upload.chunks.length} most {upload.document.predicted_label})
              </div>

              <div className="mt-3 space-y-3">
                {upload.chunks.map((c) => (
                  <button key={c.chunk_id} onClick={() => handleExplain(c.chunk_id)}
                    className="w-full text-left border rounded-lg p-3 bg-white text-gray-900 hover:bg-gray-200
                    focus:outline-none focus:ring-2 focus:ring-gray-300 focus:ring-offset-2">
                    <div className="text-sm font-mono"> chunk #{c.chunk_id} | {c.predicted_label} </div>
                    <div className="mt-1 text-sm opacity-80">{c.preview}</div>
                  </button>
                ))}
              </div>
            </div>
          </SlideAnimation> 

          {explain && !explain.error && (
            <SlideAnimation> 
              <div className="mt-6 border rounded-lg p-4">
                <div className="font-semibold">Explanation (chunk #{explain.chunk_id})</div>
                <div className="mt-2 text-sm"> Predicted: <span className="font-mono">{explain.predicted_label}</span> </div>
                <div className="mt-2 text-sm font-mono">
                  neg={explain.probabilities.negative.toFixed(3)}{" "}
                  neu={explain.probabilities.neutral.toFixed(3)}{" "}
                  pos={explain.probabilities.positive.toFixed(3)}
                </div>

                <div className="mt-4 text-sm font-semibold">Top contributing words</div>
                <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {explain.top_word_contributions.map(([word, score], i) => (
                    <div key={i} className="border rounded p-2 text-sm font-mono">
                      {word} <span className="opacity-70">({score.toFixed(4)})</span>
                    </div>
                  ))}
                </div>
              </div>
            </SlideAnimation> 
          )}
        </div>
      )}
    </div>
  );
}

export default DisplayFrontPage;