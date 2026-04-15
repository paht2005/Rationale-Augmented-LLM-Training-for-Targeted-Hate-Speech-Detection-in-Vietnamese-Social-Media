import time
import uuid
from urllib.parse import parse_qs, urlparse
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import (
    AnalyzeRequest, AnalyzeResponse,
    BatchAnalyzeRequest, BatchAnalyzeResponse,
    LabelScore
)
from .model import predict_label, model_info
from .highlight import load_keywords, build_lexicon_spans
from .youtube import fetch_youtube_comments

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

@app.get("/")
def root():
    return {"message": "Backend is running"}

def normalize_video_id(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    if "youtube.com" in raw or "youtu.be" in raw or raw.startswith("http"):
        parsed = urlparse(raw)
        if "youtu.be" in parsed.netloc:
            vid = parsed.path.lstrip("/")
            return vid or raw
        qs = parse_qs(parsed.query)
        vid = qs.get("v", [None])[0]
        if vid:
            return vid
    return raw

# CORS
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if not origins:
    raise RuntimeError("CORS_ORIGINS must be set to at least one origin. Do not use wildcard '*' in production.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load keywords once at startup
try:
    KEYWORDS = load_keywords(settings.KEYWORDS_PATH)
except Exception as e:
    KEYWORDS = []
    print(f"[WARN] Cannot load keywords from {settings.KEYWORDS_PATH}: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "version": settings.VERSION, "keywords_loaded": len(KEYWORDS)}

@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    label, score, dist = predict_label(req.text, threshold=req.options.threshold)
    labels = [LabelScore(name=n, score=s) for n, s in dist]

    highlights = []
    if req.options.return_spans:
        highlights = build_lexicon_spans(req.text, KEYWORDS)

    latency_ms = int((time.perf_counter() - start) * 1000)

    return AnalyzeResponse(
        request_id=request_id,
        label=label,
        score=score,
        labels=labels,
        highlights=highlights,
        meta={
            "model": model_info(),
            "latency_ms": latency_ms
        }
    )

@app.post("/v1/analyze/batch", response_model=BatchAnalyzeResponse)
def analyze_batch(req: BatchAnalyzeRequest):
    batch_id = str(uuid.uuid4())
    results = []
    for t in req.texts:
        single = AnalyzeRequest(text=t, lang=req.lang, options=req.options)
        results.append(analyze(single))
    return BatchAnalyzeResponse(
        request_id=batch_id,
        results=results,
        meta={"count": len(results)}
    )

@app.get("/v1/youtube/comments")
async def youtube_comments(
    video_id: str = Query(..., min_length=5),
    max_results: int = Query(50, ge=1, le=100),
    page_token: str | None = None
):
    if not settings.YOUTUBE_API_KEY:
        raise HTTPException(status_code=400, detail="Missing YOUTUBE_API_KEY in environment.")

    normalized_video_id = normalize_video_id(video_id)
    if len(normalized_video_id) < 5:
        raise HTTPException(status_code=400, detail="Invalid video_id.")

    try:
        data = await fetch_youtube_comments(
            api_key=settings.YOUTUBE_API_KEY,
            video_id=normalized_video_id,
            max_results=max_results,
            page_token=page_token
        )
        return data
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Failed to reach YouTube API.") from exc
