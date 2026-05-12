import os
import uuid
import shutil
import json
import numpy as np
import librosa
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import ProcessAudioArray
import VocalsExtractor
import dtw

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 22050
MAX_KEY_OFFSET = 400
PHRASE_VS_KEY_RATIO = 0.75
MIN_VOICED_FRAMES = 10
SNR_THRESHOLD = 1.75

sessions = {}

app.mount("/static", StaticFiles(directory="static"), name="static")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def to_json(obj):
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


def process_reference(ref_path):
    refVocals, snr = VocalsExtractor.extractVocals(ref_path, SAMPLE_RATE)
    processed = ProcessAudioArray.SwiftProcessMethod(refVocals)
    cents = processed["cents"]
    medianCents = float(np.nanmedian(cents))
    normalizedCents = cents - medianCents
    return {
        "FileName": os.path.basename(ref_path),
        "phraseNumber": 1,
        "cents": normalizedCents,
        "medianCents": medianCents,
        "voiced": processed["voiced"],
        "confidence": processed["confidence"]
    }


def process_candidate(cand_path):
    audioFileName = os.path.basename(cand_path)
    vocalsOnly, snr = VocalsExtractor.extractVocals(cand_path, SAMPLE_RATE)
    processed = ProcessAudioArray.SwiftProcessMethod(vocalsOnly)
    cents = processed["cents"]
    medianCents = float(np.nanmedian(cents))
    normalizedCents = cents - medianCents
    return {
        "FileName": audioFileName,
        "phraseNumber": 1,
        "cents": normalizedCents,
        "snr": snr,
        "medianCents": medianCents,
        "voiced": processed["voiced"],
        "confidence": processed["confidence"]
    }


def score_candidate(refData, candData):
    candCents = candData["cents"]
    voicedCount = int(np.sum(~np.isnan(candCents)))
    snr = candData["snr"]
    audioFileName = candData["FileName"]

    if voicedCount < MIN_VOICED_FRAMES:
        return {
            "track_id": audioFileName,
            "snr": round(float(snr), 4) if snr else None,
            "status": "skipped",
            "score": None,
            "dtw_score": None,
            "key_offset_cents": None,
            "key_penalty": None,
            "snr_penalty": None,
            "reason": f"only {voicedCount} voiced frames"
        }

    dtwScore = dtw.calcDTW(refData, candData)

    if dtwScore is None:
        return {
            "track_id": audioFileName,
            "snr": round(float(snr), 4) if snr else None,
            "status": "skipped",
            "score": None,
            "dtw_score": None,
            "key_offset_cents": None,
            "key_penalty": None,
            "snr_penalty": None,
            "reason": "DTW could not be computed"
        }

    refMedianCents = float(refData["medianCents"])
    candMedianCents = float(candData["medianCents"])
    keyOffset = abs(refMedianCents - candMedianCents)
    keyPenalty = max(0.0, 100.0 * (1.0 - keyOffset / MAX_KEY_OFFSET))

    phraseScore = round(PHRASE_VS_KEY_RATIO * dtwScore + (1 - PHRASE_VS_KEY_RATIO) * keyPenalty, 2)

    snrPenalty = min(1.0, snr / SNR_THRESHOLD) if snr else 1.0
    phraseScore = phraseScore * snrPenalty

    return {
        "track_id": audioFileName,
        "snr": round(float(snr), 4) if snr else None,
        "status": "scored",
        "score": round(float(phraseScore), 2),
        "dtw_score": round(float(dtwScore), 2),
        "key_offset_cents": round(float(keyOffset), 2),
        "key_penalty": round(float(keyPenalty), 2),
        "snr_penalty": round(float(snrPenalty), 4),
        "reason": None
    }


@app.post("/api/upload")
async def upload_files(
    reference: UploadFile = File(...),
    candidates: list[UploadFile] = File(...)
):
    if len(candidates) > 15:
        raise HTTPException(status_code=400, detail="Maximum 15 candidate tracks allowed")

    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True)

    ref_path = session_dir / reference.filename
    with open(ref_path, "wb") as f:
        f.write(await reference.read())

    cand_paths = []
    for cand in candidates:
        cand_path = session_dir / cand.filename
        with open(cand_path, "wb") as f:
            f.write(await cand.read())
        cand_paths.append(str(cand_path))

    sessions[session_id] = {
        "ref_path": str(ref_path),
        "cand_paths": cand_paths,
        "session_dir": str(session_dir)
    }

    return {"session_id": session_id, "candidate_count": len(cand_paths)}


@app.get("/api/process/{session_id}")
def process_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    ref_path = session["ref_path"]
    cand_paths = session["cand_paths"]
    session_dir = session["session_dir"]
    total = len(cand_paths)

    def event_stream():
        try:
            yield sse_event("log", {"message": "Loading reference audio...", "progress": 0, "current": 0, "total": total})

            refData = process_reference(ref_path)

            yield sse_event("log", {"message": "Reference audio processed", "progress": 5, "current": 0, "total": total})

            results = []
            for i, cand_path in enumerate(cand_paths):
                filename = os.path.basename(cand_path)

                yield sse_event("log", {
                    "message": f"Extracting vocals from {filename}...",
                    "progress": 5 + int((i / total) * 90),
                    "current": i,
                    "total": total
                })

                candData = process_candidate(cand_path)

                yield sse_event("log", {
                    "message": f"Scoring {filename}...",
                    "progress": 5 + int(((i + 0.7) / total) * 90),
                    "current": i,
                    "total": total
                })

                result = score_candidate(refData, candData)
                results.append(result)

                yield sse_event("log", {
                    "message": f"Completed {filename}",
                    "progress": 5 + int(((i + 1) / total) * 90),
                    "current": i + 1,
                    "total": total
                })

            scored = [r for r in results if r["score"] is not None]
            unscored = [r for r in results if r["score"] is None]
            scored.sort(key=lambda r: r["score"], reverse=True)

            ranking = []
            for rank, r in enumerate(scored, start=1):
                r["rank"] = rank
                ranking.append(r)
            for r in unscored:
                r["rank"] = None
                ranking.append(r)

            yield sse_event("log", {"message": "Ranking complete!", "progress": 100, "current": total, "total": total})
            yield sse_event("result", to_json({"ranking": ranking}))

        finally:
            del sessions[session_id]
            shutil.rmtree(session_dir, ignore_errors=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def sse_event(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data, cls=NumpyEncoder)}\n\n"


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
