"""LinkedIn job scraping via Scrapling's logged-out (guest) endpoints.

No login, no cookies. Two guest surfaces LinkedIn serves to anonymous clients:
  - search:  /jobs-guest/jobs/api/seeMoreJobPostings/search  (HTML list of cards)
  - detail:  /jobs-guest/jobs/api/jobPosting/<id>            (HTML card fragment)

Clean fetch<->parse seam: `parse_search`/`parse_job` are pure (Selector in,
model out) and unit-tested offline against saved HTML; `fetch` is the only part
that touches the network. Required fields absent => `ScrapeError` (markup drift
or a login wall), never a silently-empty JobPosting.
"""

import os
import re
import time
import logging
import itertools
from urllib.parse import urlparse, quote_plus

from scrapling.parser import Selector

from models import JobPosting

logger = logging.getLogger("linkedin")

SEARCH_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
GUEST_JOB_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
JOB_VIEW_URL = "https://www.linkedin.com/jobs/view/{job_id}"

SCRAPE_TIMEOUT = 20  # seconds per request
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds; *2^attempt

# ponytail: single-IP scraping caps out fast — LinkedIn 429s by IP, not by
# request count, so raising concurrency alone just hits the wall harder. Set
# LINKEDIN_PROXIES=url1,url2,... (http://user:pass@host:port) to spread requests
# across egress IPs — *that's* what lets you scrape more. Empty = direct
# connection (unchanged). Round-robin cycle; next() is atomic under the GIL, so
# no lock. Per-proxy health/eviction only if a dead proxy in the pool bites.
_PROXIES = [p.strip() for p in os.getenv("LINKEDIN_PROXIES", "").split(",") if p.strip()]
_proxy_cycle = itertools.cycle(_PROXIES) if _PROXIES else None


def _next_proxy() -> str | None:
    return next(_proxy_cycle) if _proxy_cycle else None


class ScrapeError(Exception):
    """Raised when scraping fails; carries the API error code + HTTP status."""

    def __init__(self, code: str, detail: str, status: int = 502):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status


def job_id_from_url(url: str) -> str:
    """Pull the numeric job id from any LinkedIn job URL.

    Handles `/jobs/view/<slug>-<id>`, `/jobs/view/<id>`, and `?currentJobId=<id>`.
    """
    m = re.search(r"currentJobId=(\d{5,})", url)
    if m:
        return m.group(1)
    last = urlparse(url).path.rstrip("/").split("/")[-1]
    m = re.search(r"(\d{5,})$", last)  # trailing digits of the last path segment
    if not m:
        raise ScrapeError("invalid_url", f"No job id found in URL: {url}", status=400)
    return m.group(1)


# --- pure parsers (no network; unit-tested against fixtures) -----------------


def _text(page, *selectors) -> str | None:
    for sel in selectors:
        val = page.css(f"{sel}::text").get()
        if val and val.strip():
            return val.strip()
    return None


def _all_text(page, *selectors) -> str | None:
    """First selector that yields any descendant text, joined and collapsed."""
    for sel in selectors:
        parts = page.css(f"{sel} ::text").getall()
        text = " ".join(p.strip() for p in parts if p and p.strip())
        if text:
            return text
    return None


def parse_search(page: Selector) -> list[JobPosting]:
    """Parse the guest search response (a list of job cards)."""
    postings: list[JobPosting] = []
    cards = page.css("div.base-card") or page.css("li")
    for card in cards:
        url = (
            card.css("a.base-card__full-link::attr(href)").get()
            or card.css("a::attr(href)").get()
        )
        title = _text(card, "h3.base-search-card__title", "h3")
        company = _text(card, "h4.base-search-card__subtitle a", "h4 a", "h4")
        if not (url and title and company):
            continue  # skip promo/empty cards rather than emit junk
        try:
            job_id = job_id_from_url(url)
        except ScrapeError:
            continue
        postings.append(
            JobPosting(
                linkedin_job_id=job_id,
                url=url.split("?")[0],
                title=title,
                company=company,
                location=_text(
                    card, ".job-search-card__location", ".job-result-card__location"
                ),
                posted_date=card.css("time::attr(datetime)").get(),
            )
        )
    if not postings and cards:
        raise ScrapeError("scrape_failed", "Search markup changed; parsed 0 postings")
    return postings


def parse_job(page: Selector, url: str) -> JobPosting:
    """Parse a single job detail page/fragment."""
    title = _text(
        page,
        "h1.top-card-layout__title",
        ".topcard__title",
        "h1",
        "h2.top-card-layout__title",
    )
    company = _text(
        page,
        "a.topcard__org-name-link",
        ".top-card-layout__second-subline a",
        ".topcard__flavor",
    )
    if not (title and company):
        # Missing the essentials => markup drift or a login wall was served.
        raise ScrapeError(
            "scrape_failed", "Could not parse title/company from job page"
        )

    seniority = employment_type = None
    for item in page.css(".description__job-criteria-item"):
        label = (_text(item, ".description__job-criteria-subheader") or "").lower()
        value = _text(item, ".description__job-criteria-text")
        if "seniority" in label:
            seniority = value
        elif "employment" in label:
            employment_type = value

    return JobPosting(
        linkedin_job_id=job_id_from_url(url),
        url=url.split("?")[0],
        title=title,
        company=company,
        location=_text(
            page,
            ".topcard__flavor--bullet",
            ".top-card-layout__second-subline .topcard__flavor",
        ),
        posted_date=_text(page, ".posted-time-ago__text", "time"),
        description=_all_text(
            page,
            ".show-more-less-html__markup",
            ".description__text",
            ".decorated-job-posting__details",
        ),
        seniority=seniority,
        employment_type=employment_type,
    )


# --- network (the only part that hits LinkedIn) ------------------------------


class LinkedInScraper:
    """Stateless scraper. One instance is shared; holds no per-request state."""

    def fetch(self, url: str) -> Selector:
        """GET `url` via Scrapling's HTTP Fetcher, retrying on 429/5xx.

        ponytail: Fetcher (curl_cffi TLS impersonation, no browser). If LinkedIn
        starts blocking this, switch to StealthyFetcher (Camoufox) — needs
        `pip install "scrapling[fetchers]"` + `scrapling install` for the browser.
        Rotates egress IP per request when LINKEDIN_PROXIES is set (see above).
        """
        from scrapling.fetchers import Fetcher  # heavy import; only on real fetch

        last_status = None
        for attempt in range(_MAX_RETRIES):
            proxy = _next_proxy()
            resp = Fetcher.get(
                url,
                stealthy_headers=True,
                timeout=SCRAPE_TIMEOUT,
                **({"proxy": proxy} if proxy else {}),
            )
            status = getattr(resp, "status", 200)
            last_status = status
            if status == 404:
                raise ScrapeError(
                    "job_not_found", f"LinkedIn returned 404 for {url}", status=404
                )
            if status < 400:
                return resp
            if status not in (429, 500, 502, 503, 504):
                break
            time.sleep(_BACKOFF_BASE * (2**attempt))  # transient; back off and retry
        raise ScrapeError(
            "scrape_failed",
            f"LinkedIn returned HTTP {last_status} for {url}",
            status=502,
        )

    def fetch_job(self, url_or_id: str) -> JobPosting:
        job_id = url_or_id if url_or_id.isdigit() else job_id_from_url(url_or_id)
        page = self.fetch(GUEST_JOB_URL.format(job_id=job_id))
        return parse_job(page, url=JOB_VIEW_URL.format(job_id=job_id))

    def search(
        self, keywords: str, location: str = "", start: int = 0
    ) -> list[JobPosting]:
        url = f"{SEARCH_URL}?keywords={quote_plus(keywords)}&location={quote_plus(location)}&start={start}"
        return parse_search(self.fetch(url))
