from __future__ import annotations

from .base import Agent

LEVEL_FACTOR = {"advanced": 1.0, "intermediate": 0.85, "beginner": 0.6}


def readiness_score(rows: list[dict]) -> int:
    """Importance-weighted share of market skills the student covers (0-100)."""
    total = sum(r["importance"] for r in rows) or 1
    got = 0.0
    for r in rows:
        if r["status"] == "have":
            got += r["importance"] * LEVEL_FACTOR.get(r.get("your_level") or "beginner", 0.6)
        elif r["status"] == "partial":
            got += r["importance"] * 0.4
    return round(100 * got / total)


def priority(importance: int) -> str:
    return "high" if importance >= 50 else "medium" if importance >= 30 else "low"


class SkillGapAgent(Agent):
    name = "Skill-Gap Agent"
    goal = "compare the student's skills against what the job market asks for, decide which requirements are covered, and explain the gaps."

    def run(self, state: dict) -> dict:
        prof, market = state["profile"], state["market"]
        with self.step(state, "Match profile against market requirements") as log:
            system = self.system(
                "For EVERY market skill decide if the student covers it, using meaning, not exact spelling "
                "(e.g. 'Spring Boot' covers 'Backend framework'; 'MySQL' covers 'SQL'; 'pandas' partly covers 'Python').\n"
                "status: have (clearly covered) | partial (related or shallow evidence) | missing.\n"
                "covered_by: the student's skill(s) that cover it, or null. "
                "reason: one short line; for missing skills say why it matters for this role.\n"
                "Then give 'advice': 2 sentences on what to do first.\n"
                "JSON schema: {\"rows\": [{\"skill\": str, \"status\": \"have|partial|missing\", "
                "\"covered_by\": str|null, \"reason\": str}], \"advice\": str}"
            )
            user = (f"STUDENT SKILLS (with levels):\n{self.dump(prof['skills'])}\n\n"
                    f"STUDENT PROJECTS:\n{self.dump(prof.get('projects', []), 2500)}\n\n"
                    f"MARKET SKILLS for {market['role']}:\n"
                    f"{self.dump([{'skill': s['name'], 'importance': s['importance']} for s in market['skills']])}")
            data = self.llm.json(system, user, required=("rows",))

            by_name = {str(r.get("skill", "")).strip().lower(): r for r in data.get("rows") or []}
            levels = {s["name"].lower(): s["level"] for s in prof["skills"]}
            rows = []
            for s in market["skills"]:
                r = by_name.get(s["name"].lower(), {})
                status = r.get("status") if r.get("status") in ("have", "partial", "missing") else "missing"
                covered = r.get("covered_by")
                lvl = None
                if covered:
                    # take the best level among the covering skills
                    cand = [levels.get(c.strip().lower()) for c in str(covered).replace("/", ",").split(",")]
                    cand = [c for c in cand if c]
                    lvl = max(cand, key=lambda x: LEVEL_FACTOR[x]) if cand else "intermediate"
                rows.append({"skill": s["name"], "importance": s["importance"], "status": status,
                             "covered_by": covered, "your_level": lvl if status == "have" else None,
                             "reason": r.get("reason", s.get("why", "")), "priority": priority(s["importance"])})

            score = readiness_score(rows)
            gaps = [r for r in rows if r["status"] != "have"]
            gaps.sort(key=lambda r: (r["status"] == "partial", -r["importance"]))
            verdict = ("Strong match: focus on projects and mock interviews, then apply." if score >= 85 else
                       "Good base: close the high-priority gaps and you will clear most job descriptions." if score >= 60 else
                       "Developing: follow the roadmap in order; early weeks cover the most-asked skills." if score >= 35 else
                       "Early stage: start with the top three high-priority skills before anything else.")
            gap = {"rows": rows, "gaps": gaps, "readiness": score, "verdict": verdict,
                   "advice": data.get("advice", "")}
            state["gap"] = gap
            state.setdefault("readiness_history", []).append(score)
            log["detail"] = (f"readiness {score}/100 · {sum(r['status']=='have' for r in rows)} have, "
                             f"{sum(r['status']=='partial' for r in rows)} partial, "
                             f"{sum(r['status']=='missing' for r in rows)} missing")
        return gap
