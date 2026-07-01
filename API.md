# Resume Extraction API

HTTP wrapper around the PDF → structured-JSON → score pipeline. Built to be
called server-side by `rebase-resume` (Next.js). Internal service — **no
app-layer auth** (front it with a gateway); input is still validated.

Source: [`app.py`](app.py). Interactive docs at `GET /docs` (OpenAPI).

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000   # or: python app.py
```

Docker:

```bash
docker build -t hiring-agent-api .
docker run -p 8000:8000 -e GEMINI_API_KEY=... hiring-agent-api
```

### Environment

| Var | Default | Notes |
|-----|---------|-------|
| `LLM_PROVIDER` | `ollama` | Set to `gemini` for prod |
| `DEFAULT_MODEL` | `gemma3:4b` | Prod: `gemini-2.5-flash` |
| `GEMINI_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `GITHUB_TOKEN` | — | Optional; raises GitHub rate limit for `/score` & `/process` |
| `PORT` | `8000` | |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `MAX_PAGES` | `15` | Page-count limit |
| `MAX_CONCURRENT_EXTRACTIONS` | `4` | Global cap on in-flight LLM requests (extract/score/match) |
| `MAX_CONCURRENT_SCRAPE` | `2` | Global cap on in-flight LinkedIn scrapes (be polite; avoid 429s) |

## Endpoints

All responses are JSON. Every response carries an `X-Request-ID` header.

### `GET /health`
Liveness. → `200 {"status":"ok"}`.

### `GET /ready`
Readiness — 503 if `LLM_PROVIDER=gemini` but no key.
→ `200 {"status":"ready","provider":"gemini","model":"gemini-2.5-flash"}`.

### `POST /extract`
PDF → structured resume. **The core endpoint.**

- Body: `multipart/form-data`, field `file` = the PDF.
- Query: `include_markdown` (bool, default `false`).
- → `200` with a **`JSONResume`** object (see schema below).
- With `?include_markdown=true` → `{"resume": JSONResume, "markdown": "..."}`.

```bash
curl -F "file=@resume.pdf" http://localhost:8000/extract
```

Latency ~10–20s cold (6 section LLM calls run in parallel). Synchronous — the
caller holds the request; there is no job/polling API.

### `POST /score`
Structured resume → evaluation. Enriches with GitHub when `basics.profiles`
contains a GitHub profile.

- Body: `application/json`, a **`JSONResume`** object.
- → `200` with an **`EvaluationData`** object.

```bash
curl -H "Content-Type: application/json" -d @resume.json http://localhost:8000/score
```

### `POST /process`
One-shot: PDF → extract → GitHub enrich → score.

- Body: `multipart/form-data`, field `file` = the PDF.
- → `200 {"resume": JSONResume, "evaluation": EvaluationData}`.

```bash
curl -F "file=@resume.pdf" http://localhost:8000/process
```

### `POST /scrape-job`
Scrape one LinkedIn job posting from the **logged-out (guest) endpoints** — no
auth, no cookies.

- Body: `application/json`, `{ "url": "https://www.linkedin.com/jobs/view/<id>" }`
  or `{ "job_id": "<numeric id>" }`. When `url` is given, its host must be
  `linkedin.com`.
- → `200` with a **`JobPosting`** object.

```bash
curl -H "Content-Type: application/json" \
  -d '{"url":"https://www.linkedin.com/jobs/view/3812345678"}' \
  http://localhost:8000/scrape-job
```

### `GET /search-jobs`
Search jobs via the guest search endpoint. Returns summary cards (no full
description — call `/scrape-job` for the details of a specific posting).

- Query: `keywords` (required), `location` (default `""`), `start` (default `0`, page offset).
- → `200` with a list of **`JobPosting`** objects (`description`/`seniority`/`employment_type` are `null`).

```bash
curl "http://localhost:8000/search-jobs?keywords=backend%20engineer&location=Remote"
```

### `POST /match`
Score how well a resume fits a job description (single LLM call). Pair with
`/extract` (resume → `JSONResume`) and `/scrape-job` (job → `description`).

- Body: `application/json`, `{ "resume": JSONResume, "job_description": "..." }`.
- → `200` with a **`JobMatch`** object.

```bash
curl -H "Content-Type: application/json" \
  -d '{"resume": {"basics": {"name": "Jane"}}, "job_description": "..."}' \
  http://localhost:8000/match
```

> **Scraping is best-effort.** LinkedIn changes markup and rate-limits bursts.
> `scrape_failed` means the markup drifted or a login wall was served; empty
> fields or repeated 429s are the signal to add proxies / switch Scrapling to
> `StealthyFetcher` (see [`linkedin.py`](linkedin.py)).

## Errors

Non-2xx responses share this shape:

```json
{ "error": "not_a_pdf", "detail": "File is not a PDF (missing %PDF header)", "request_id": "..." }
```

| Status | `error` | When |
|--------|---------|------|
| 400 | `invalid_content_type` | `file` isn't `application/pdf` |
| 400 | `empty_file` | zero bytes |
| 400 | `not_a_pdf` | missing `%PDF` magic bytes |
| 400 | `unreadable_pdf` | PyMuPDF can't open it |
| 400 | `too_many_pages` | exceeds `MAX_PAGES` |
| 413 | `file_too_large` | exceeds `MAX_UPLOAD_MB` |
| 422 | `extraction_failed` | pipeline produced no usable resume (e.g. LLM/quota failure during extraction) |
| 502 | `scoring_failed` | evaluation LLM call failed |
| 400 | `invalid_url` | `/scrape-job` given no `url`/`job_id`, or a non-`linkedin.com` URL |
| 404 | `job_not_found` | LinkedIn returned 404 for the job |
| 502 | `scrape_failed` | markup drift / login wall / repeated 429–5xx from LinkedIn |
| 502 | `match_failed` | `/match` LLM call failed |
| 500 | `internal_error` | unexpected |

## Response schemas

Both are Pydantic models in [`models.py`](models.py); the OpenAPI schema at
`/docs` is authoritative. All `JSONResume` fields are optional (a section the
resume lacks is `null`).

### `JSONResume`

```jsonc
{
  "basics":   { "name": "Jane Doe", "email": "...", "phone": "...", "url": "...",
                "summary": "...", "location": { "city": "...", "region": "...", ... },
                "profiles": [ { "network": "Github", "username": "...", "url": "..." } ] },
  "work":     [ { "name": "Acme", "position": "...", "startDate": "...", "endDate": "...",
                  "summary": "...", "highlights": ["..."] } ],
  "education":[ { "institution": "...", "area": "...", "studyType": "...",
                  "startDate": "...", "endDate": "...", "score": "...", "courses": ["..."] } ],
  "skills":   [ { "name": "Python", "level": "...", "keywords": ["..."] } ],
  "projects": [ { "name": "...", "description": "...", "url": "...",
                  "technologies": ["..."], "highlights": ["..."] } ],
  "awards":   [ { "title": "...", "date": "...", "awarder": "...", "summary": "..." } ],
  // also (usually null): volunteer, certificates, publications, languages, interests, references
}
```

### `EvaluationData`

```jsonc
{
  "scores": {
    "open_source":     { "score": 0, "max": 30, "evidence": "..." },
    "self_projects":   { "score": 0, "max": 20, "evidence": "..." },
    "production":      { "score": 0, "max": 30, "evidence": "..." },
    "technical_skills":{ "score": 0, "max": 20, "evidence": "..." }
  },
  "bonus_points":  { "total": 0, "breakdown": "..." },   // total 0–20
  "deductions":    { "total": 0, "reasons": "..." },      // positive, applied as negative
  "key_strengths":         ["...", "..."],   // 1–5 items
  "areas_for_improvement": ["...", "..."]    // 1–5 items
}
```

Final score is computed from these by the caller (see `evaluator.py` constants:
`MIN_FINAL_SCORE=-20`, `MAX_FINAL_SCORE=120`, `MAX_BONUS_POINTS=20`).

### `JobPosting`

```jsonc
{
  "linkedin_job_id": "3812345678",
  "url": "https://www.linkedin.com/jobs/view/3812345678",
  "title": "Backend Engineer",
  "company": "Acme Corp",
  "location": "Remote, United States",   // nullable
  "posted_date": "2026-06-25",           // nullable (ISO date or "1 week ago")
  "description": "...",                   // nullable; null in /search-jobs results
  "seniority": "Mid-Senior level",        // nullable
  "employment_type": "Full-time"          // nullable
}
```

### `JobMatch`

```jsonc
{
  "fit_score": 82,                 // 0–100
  "strengths": ["...", "..."],     // 1–5 items
  "gaps": ["...", "..."],          // 1–5 items
  "summary": "One paragraph explaining the score."
}
```

## Notes for the next agent

- **Wiring rebase-resume:** call `POST /process` from a Server Action with the
  uploaded file; render `evaluation` in place of the hardcoded `61` in
  [`UploadFlow.tsx`](../rebase-resume/components/organisms/UploadFlow.tsx). Add
  an `EXTRACTION_API_URL` env var. Server-side call → no CORS needed.
- **Sync by design.** No job queue. If you need progressive UI, keep the
  "analyzing" screen and await the single response, or add async later.
- The dev file-cache (`cache/`), CSV export, and CLI (`score.py`) are **not**
  used by the API path — don't rely on them.
- Tests: `pytest tests/test_api.py` (LLM mocked, no cost/network).
