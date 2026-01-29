import numpy as np
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from pydantic import BaseModel
from app import text_processing as tp

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, List[Dict[str, Any]]] = {}

@app.post("/upload-text")
async def upload_text(file: UploadFile = File(...), top_n: int = 20):
    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    chunks = tp.roberta_chunk_text(text)
    if not chunks:
        return {"session_id": None, "chunks": [], "message": "No text found."}

    probabilities = tp.roberta_predict_probabilities(chunks) 
    doc_probabilities = probabilities.mean(axis=0)
    doc_prediction = int(np.argmax(doc_probabilities))
    doc_label = tp.LABELS[doc_prediction]

    ranked_chunk_indices = np.argsort(-probabilities[:, doc_prediction]) 
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = [{"chunk_id": i, "text": chunks[i]} for i in range(len(chunks))]

    chunk_summaries = []
    for chunk_index in ranked_chunk_indices[: min(top_n, len(chunks))]:
        chunk_preview = chunks[chunk_index].strip().replace("\n", " ")
        chunk_preview = chunk_preview[:160] + ("…" if len(chunk_preview) > 160 else "")
        chunk_summaries.append({
            "chunk_id": int(chunk_index),
            "preview": chunk_preview,
            "predicted_label": tp.LABELS[int(np.argmax(probabilities[chunk_index]))],
            "probabilities": {
                "negative": float(probabilities[chunk_index, 0]),
                "neutral": float(probabilities[chunk_index, 1]),
                "positive": float(probabilities[chunk_index, 2]),
            }
        })

    return {
        "session_id": session_id,
        "document": {
            "predicted_label": doc_label,
            "probabilities": {
                "negative": float(doc_probabilities[0]),
                "neutral": float(doc_probabilities[1]),
                "positive": float(doc_probabilities[2]),
            },
            "num_chunks": int(len(chunks)),
        },
        "chunks": chunk_summaries,
    }


class ExplainRequest(BaseModel):
    session_id: str
    chunk_id: int
    top_contribution_words: int = 15

@app.post("/explain-chunk")
async def explain_chunk(req: ExplainRequest):
    session_chunks = SESSIONS.get(req.session_id)
    if not session_chunks:
        return {"error": "Unknown session_id."}

    if req.chunk_id < 0 or req.chunk_id >= len(session_chunks):
        return {"error": "Invalid chunk_id."}

    text = session_chunks[req.chunk_id]["text"]
    result = tp.explain_prediction(text, num_of_top_words=req.top_contribution_words)

    return {
        "chunk_id": req.chunk_id,
        **result
    }