"""Tool: fetch live job postings from the Adzuna API (free developer key).

https://developer.adzuna.com/ — India is country code "in".
If no key is configured the Job-Market Agent falls back to the curated baseline in
data/role_baseline.json plus the LLM's own knowledge, and says so in its output.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"


def search_jobs(role: str, app_id: str, app_key: str, location: str = "", country: str = "in", n: int = 25):
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": role,
        "results_per_page": n,
        "content-type": "application/json",
    }
    if location:
        params["where"] = location
    url = ADZUNA_URL.format(country=country) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "CareerPilot-AI/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    postings = []
    for j in data.get("results", []):
        postings.append(
            {
                "title": j.get("title", ""),
                "company": (j.get("company") or {}).get("display_name", ""),
                "location": (j.get("location") or {}).get("display_name", ""),
                "description": re.sub(r"<[^>]+>", " ", j.get("description", "")),
                "url": j.get("redirect_url", ""),
            }
        )
    return postings


def skill_frequency(skills: list[str], postings: list[dict], synonyms: dict[str, list[str]] | None = None):
    """Share of postings (0-100) whose title+description mentions each skill or a synonym."""
    synonyms = synonyms or {}
    if not postings:
        return {}
    texts = [(p["title"] + " " + p["description"]).lower() for p in postings]
    out = {}
    for s in skills:
        terms = [s.lower()] + [t.lower() for t in synonyms.get(s, [])]
        pats = [re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])") for t in terms if t]
        hits = sum(1 for tx in texts if any(p.search(tx) for p in pats))
        out[s] = round(100 * hits / len(texts))
    return out
