"""Production HTTP API wrapping the resume extraction + scoring pipeline.

Endpoints:
  POST /extract  (multipart pdf)          -> JSONResume        (optional ?include_markdown=true)
  POST /score    (JSON JSONResume body)   -> EvaluationData
  POST /process     (multipart pdf)       -> {resume, evaluation}
  POST /scrape-job  (JSON {url|job_id})   -> JobPosting        (LinkedIn, logged-out)
  GET  /search-jobs (?keywords=&location=)-> [JobPosting]
  POST /match       (JSON {resume, job_description}) -> JobMatch
  POST /generate    (JSON {resume, job_description, outputs}) -> GenerateResult
  GET  /health                            -> liveness
  GET  /ready                             -> readiness (provider configured)

Internal service: no app-layer auth (front it with a gateway). Input is still
validated (content-type, size, %PDF magic, page count).
"""

import os
import time
import uuid
import asyncio
import logging

import pymupdf
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from urllib.parse import urlparse

from pdf import PDFHandler
from models import JSONResume, EvaluationData, JobPosting, JobMatch, GenerateResult
from prompt import DEFAULT_MODEL, GEMINI_API_KEY, PROVIDER
from score import _evaluate_resume, find_profile
from github import fetch_and_display_github_info
from linkedin import LinkedInScraper, ScrapeError, JOB_VIEW_URL
from matcher import JobMatcher
from generator import DocumentGenerator

logger = logging.getLogger("api")

# --- config via env ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "15"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_EXTRACTIONS", "4"))
MAX_CONCURRENT_SCRAPE = int(os.getenv("MAX_CONCURRENT_SCRAPE", "2"))

# ponytail: one global semaphore caps concurrent LLM-bound requests so we don't
# blow Gemini quota. Per-key limits only if this ever goes multi-tenant.
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# ponytail: separate, tighter cap for scraping — it's network I/O against
# LinkedIn (not LLM-bound) and being polite avoids rate-limit blocks. Merge with
# _semaphore only if they ever need a shared limit.
_scrape_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPE)

_handler: PDFHandler | None = None
_scraper: LinkedInScraper | None = None
_matcher: JobMatcher | None = None
_generator: DocumentGenerator | None = None


def get_handler() -> PDFHandler:
    """Lazily build one shared PDFHandler (it initializes the LLM provider)."""
    global _handler
    if _handler is None:
        _handler = PDFHandler()
    return _handler


def get_scraper() -> LinkedInScraper:
    global _scraper
    if _scraper is None:
        _scraper = LinkedInScraper()
    return _scraper


def get_matcher() -> JobMatcher:
    """Lazily build one shared JobMatcher (it initializes the LLM provider)."""
    global _matcher
    if _matcher is None:
        _matcher = JobMatcher()
    return _matcher


def get_generator() -> DocumentGenerator:
    """Lazily build one shared DocumentGenerator (it initializes the LLM provider)."""
    global _generator
    if _generator is None:
        _generator = DocumentGenerator()
    return _generator


app = FastAPI(title="Resume Extraction API", version="1.0.0")


class ApiError(Exception):
    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error,
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "Unexpected server error",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.middleware("http")
async def _request_context(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info(
        "%s %s -> %s %.0fms rid=%s",
        request.method,
        request.url.path,
        response.status_code,
        dur_ms,
        request.state.request_id,
    )
    return response


async def _read_validated_pdf(file: UploadFile) -> bytes:
    if file.content_type not in ("application/pdf", "application/octet-stream", None):
        raise ApiError(
            400,
            "invalid_content_type",
            f"Expected application/pdf, got {file.content_type}",
        )
    data = await file.read()
    if not data:
        raise ApiError(400, "empty_file", "Uploaded file is empty")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ApiError(
            413,
            "file_too_large",
            f"File is {size_mb:.1f}MB; limit is {MAX_UPLOAD_MB}MB",
        )
    if not data.startswith(b"%PDF"):
        raise ApiError(400, "not_a_pdf", "File is not a PDF (missing %PDF header)")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = doc.page_count
    except Exception as e:
        raise ApiError(400, "unreadable_pdf", f"Could not open PDF: {e}")
    if pages > MAX_PAGES:
        raise ApiError(
            400, "too_many_pages", f"PDF has {pages} pages; limit is {MAX_PAGES}"
        )
    return data


def _score_resume(resume: JSONResume) -> EvaluationData:
    """Enrich with GitHub (when a profile is present) and evaluate."""
    github_data = {}
    profiles = resume.basics.profiles if resume.basics else None
    gh = find_profile(profiles, "Github")
    if gh and gh.url:
        try:
            github_data = fetch_and_display_github_info(gh.url) or {}
        except Exception as e:  # enrichment is best-effort
            logger.warning("GitHub enrichment failed: %s", e)
    return _evaluate_resume(resume, github_data)


async def _extract(data: bytes) -> JSONResume:
    async with _semaphore:
        resume = await asyncio.to_thread(get_handler().extract_json_from_pdf, data)
    if resume is None:
        raise ApiError(
            422,
            "extraction_failed",
            "Could not extract a valid resume from the PDF",
        )
    return resume


async def _score(resume: JSONResume) -> EvaluationData:
    try:
        async with _semaphore:
            evaluation = await asyncio.to_thread(_score_resume, resume)
    except ApiError:
        raise
    except Exception as e:
        logger.exception("scoring failed")
        raise ApiError(502, "scoring_failed", f"LLM scoring failed: {e}")
    if evaluation is None:
        raise ApiError(502, "scoring_failed", "LLM scoring returned no result")
    return evaluation


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ApiError(
            503,
            "not_ready",
            "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set",
        )
    return {"status": "ready", "provider": PROVIDER, "model": DEFAULT_MODEL}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    include_markdown: bool = Query(False),
):
    data = await _read_validated_pdf(file)
    if not include_markdown:
        return await _extract(data)

    markdown = await asyncio.to_thread(get_handler().extract_text_from_pdf, data)
    if not markdown:
        raise ApiError(422, "extraction_failed", "Could not extract text from the PDF")
    async with _semaphore:
        resume = await asyncio.to_thread(get_handler().extract_json_from_text, markdown)
    if resume is None:
        raise ApiError(
            422,
            "extraction_failed",
            "Could not extract a valid resume from the PDF",
        )
    return {"resume": resume, "markdown": markdown}


@app.post("/score", response_model=EvaluationData)
async def score(resume: JSONResume):
    return await _score(resume)


@app.post("/process")
async def process(file: UploadFile = File(...)):
    data = await _read_validated_pdf(file)
    resume = await _extract(data)
    evaluation = await _score(resume)
    return {"resume": resume, "evaluation": evaluation}


class ScrapeJobRequest(BaseModel):
    url: str | None = None
    job_id: str | None = None


class MatchRequest(BaseModel):
    resume: JSONResume
    job_description: str


class GenerateRequest(BaseModel):
    resume: JSONResume
    job_description: str
    # Which documents to draft: any of "cover", "statement".
    outputs: list[str] = ["cover"]
    # Fine-tune knobs (quick tailor). Unknown values fall back to defaults in
    # the generator, so callers can send freely.
    tone: str = "professional"  # professional | warm | bold
    length: str = "standard"  # concise | standard | detailed
    emphasis: list[str] = ["impact", "tech"]  # impact|tech|lead|culture|story
    note: str = ""  # free-text steer, e.g. "mention I'm relocating to SF"


def _validate_linkedin_url(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if not (host == "linkedin.com" or host.endswith(".linkedin.com")):
        raise ApiError(
            400, "invalid_url", f"URL host must be linkedin.com, got '{host or url}'"
        )


@app.post("/scrape-job", response_model=JobPosting)
async def scrape_job(req: ScrapeJobRequest):
    if req.url:
        _validate_linkedin_url(req.url)
        target = req.url
    elif req.job_id:
        target = JOB_VIEW_URL.format(job_id=req.job_id)
    else:
        raise ApiError(400, "invalid_url", "Provide either 'url' or 'job_id'")
    try:
        async with _scrape_semaphore:
            return await asyncio.to_thread(get_scraper().fetch_job, target)
    except ScrapeError as e:
        raise ApiError(e.status, e.code, e.detail)


@app.get("/search-jobs", response_model=list[JobPosting])
async def search_jobs(
    keywords: str = Query(..., min_length=1),
    location: str = Query(""),
    start: int = Query(0, ge=0),
):
    try:
        async with _scrape_semaphore:
            return await asyncio.to_thread(
                get_scraper().search, keywords, location, start
            )
    except ScrapeError as e:
        raise ApiError(e.status, e.code, e.detail)


@app.post("/match", response_model=JobMatch)
async def match(req: MatchRequest):
    try:
        async with _semaphore:  # LLM-bound -> share the extraction/scoring cap
            return await asyncio.to_thread(
                get_matcher().match, req.resume.model_dump_json(), req.job_description
            )
    except Exception as e:
        logger.exception("match failed")
        raise ApiError(502, "match_failed", f"LLM match failed: {e}")


@app.post("/generate", response_model=GenerateResult)
async def generate(req: GenerateRequest):
    outputs = [o for o in req.outputs if o in ("cover", "statement")]
    if not outputs:
        raise ApiError(
            400, "invalid_outputs", "outputs must include 'cover' and/or 'statement'"
        )
    if not (req.job_description or "").strip():
        raise ApiError(400, "empty_job_description", "job_description is required")
    try:
        async with _semaphore:  # LLM-bound -> share the extraction/scoring cap
            return await asyncio.to_thread(
                get_generator().generate,
                req.resume.model_dump_json(),
                req.job_description,
                outputs,
                req.tone,
                req.length,
                req.emphasis,
                req.note,
            )
    except Exception as e:
        logger.exception("generate failed")
        raise ApiError(502, "generation_failed", f"LLM generation failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
