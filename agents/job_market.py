from __future__ import annotations

import json
from pathlib import Path

from .base import Agent
from tools.job_search import search_jobs, skill_frequency

BASELINE = json.loads((Path(__file__).resolve().parent.parent / "data" / "role_baseline.json").read_text())


class JobMarketAgent(Agent):
    name = "Job-Market Agent"
    goal = "find out which skills employers currently ask for in the student's target role, with an importance score for each."

    def __init__(self, llm, adzuna_id: str | None = None, adzuna_key: str | None = None):
        super().__init__(llm)
        self.adzuna_id, self.adzuna_key = adzuna_id, adzuna_key

    def run(self, state: dict) -> dict:
        inp = state["inputs"]
        role, location = inp["target_role"], inp.get("location", "")
        cache = state.setdefault("market_cache", {})
        ck = f"{role}|{location}".lower()
        if ck in cache:
            with self.step(state, "Reuse market research for this role (cached)") as log:
                state["market"] = cache[ck]
                log["detail"] = f"{len(cache[ck]['skills'])} skills from earlier run"
            return cache[ck]

        postings = []
        if self.adzuna_id and self.adzuna_key:
            with self.step(state, f"Fetch live job postings for '{role}'", tool="adzuna_job_search") as log:
                try:
                    postings = search_jobs(role, self.adzuna_id, self.adzuna_key, location=location)
                    log["detail"] = f"{len(postings)} postings fetched"
                except Exception as e:  # keep going on the baseline
                    log["detail"] = f"Job API unavailable ({e}); using curated baseline"
                    postings = []

        baseline = BASELINE.get(role)
        with self.step(state, "Extract required skills and score importance",
                       tool="skill_frequency" if postings else "curated_baseline") as log:
            if postings:
                evidence = "\n\n".join(
                    f"[{i+1}] {p['title']} — {p['company']}\n{p['description'][:600]}" for i, p in enumerate(postings[:25])
                )
                src = f"LIVE JOB POSTINGS ({len(postings)}):\n{evidence}"
            else:
                src = ("No live postings available. Use your knowledge of current entry-level hiring in India"
                       + (f", grounded by this curated baseline (skill: % of job descriptions): {json.dumps(baseline)}"
                          if baseline else "") + ".")
            system = self.system(
                "List the 10-14 skills that matter most for this role at entry level. "
                "Merge duplicates (e.g. 'Java/Python' as one language skill when either is accepted). "
                "For each skill give importance 0-100 (how often / how strongly employers require it), "
                "a category (core | framework | tool | cs-fundamental | soft), one line on why it matters, "
                "and 1-4 lowercase synonyms or spellings that would appear in a job posting.\n"
                "JSON schema: {\"role_summary\": str, \"skills\": [{\"name\": str, \"importance\": int, "
                "\"category\": str, \"why\": str, \"synonyms\": [str]}]}"
            )
            data = self.llm.json(system, f"Target role: {role}. Location: {location or 'India'}.\n\n{src}",
                                 required=("skills",))
            skills = []
            for s in data.get("skills") or []:
                try:
                    imp = max(0, min(100, int(s.get("importance", 50))))
                except (TypeError, ValueError):
                    imp = 50
                skills.append({"name": str(s.get("name", "")).strip(), "importance": imp,
                               "category": s.get("category", "core"), "why": s.get("why", ""),
                               "synonyms": [str(x) for x in (s.get("synonyms") or [])][:4]})
            skills = [s for s in skills if s["name"]]

            source = "LLM knowledge" + (" + curated baseline" if baseline else "")
            if postings:
                # Replace LLM guesses with measured frequency where the skill actually appears.
                freq = skill_frequency([s["name"] for s in skills], postings,
                                       {s["name"]: s["synonyms"] for s in skills})
                for s in skills:
                    s["llm_importance"] = s["importance"]
                    s["posting_share"] = freq.get(s["name"], 0)
                    if s["posting_share"] > 0:
                        s["importance"] = round(0.6 * s["posting_share"] + 0.4 * s["importance"])
                source = f"{len(postings)} live job postings (Adzuna) + LLM extraction"
            skills.sort(key=lambda s: -s["importance"])
            market = {"role": role, "role_summary": data.get("role_summary", ""), "source": source,
                      "postings_count": len(postings), "postings": postings[:10], "skills": skills}
            log["detail"] = f"{len(skills)} skills · source: {source}"
        state["market"] = market
        cache[ck] = market
        return market
