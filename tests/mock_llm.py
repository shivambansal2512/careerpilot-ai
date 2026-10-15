"""Offline stand-in for the LLM, used only by the test suite (CAREERPILOT_MOCK=1).

It answers each agent's prompt with plausible JSON built from the prompt itself, so the
whole agent pipeline, orchestrator routing and UI can be tested without an API key.
"""
from __future__ import annotations

import json
import re


class MockLLM:
    provider = "mock"
    model = "mock-llm"
    api_key = ""
    label = "Mock LLM (tests only)"

    def __init__(self, bad_json_once: bool = False):
        self.calls = 0
        self.total_tokens = 0
        self.bad_json_once = bad_json_once

    def chat(self, system, user, temperature=0.3):
        self.calls += 1
        if self.bad_json_once:
            self.bad_json_once = False
            return "Sure! Here is the data: {not valid json"
        agent = re.search(r"AGENT_ID: (.+)", system).group(1).strip()
        return "```json\n" + json.dumps(getattr(self, "_" + agent.split()[0].lower().replace("-", "_"))(system, user)) + "\n```"

    def json(self, system, user, temperature=0.3, required=()):
        from llm import LLM

        return LLM.json(self, system, user, temperature, required)  # reuse real parsing + retry logic

    # ---- canned behaviour per agent
    def _profile(self, s, u):
        typed = u.split("SKILLS THE STUDENT TYPED:")[-1].split("\n\nTarget role")[0]
        found = [t.strip() for t in re.split(r"[,\s]+", typed) if t.strip() and t.strip().lower() != "(advanced)"]
        lvl = "advanced" if "(advanced)" in typed else None
        return {"name": "Test Student", "education": "B.Tech CSE", "year": "3rd year",
                "summary": "Third-year student with programming basics.",
                "skills": [{"name": k, "level": lvl or ("intermediate" if k in ("Java", "DSA") else "beginner"),
                            "evidence": "listed"} for k in found],
                "projects": [{"name": "Hostel Booking App", "tech": ["React", "Node.js"], "what": "booking system"}],
                "strengths": ["problem solving"]}

    def _job_market(self, s, u):
        m = re.search(r"\{.*\}", u.split("baseline")[-1], re.S) if "baseline" in u else None
        base = json.loads(m.group(0)) if m else {"DSA": 90, "SQL": 70, "Git": 60, "Docker": 40, "System Design": 35}
        return {"role_summary": "Builds and maintains server-side systems.",
                "skills": [{"name": k, "importance": v, "category": "core", "why": f"{k} is asked often",
                            "synonyms": [k.lower()]} for k, v in base.items()]}

    def _skill_gap(self, s, u):
        market = json.loads(re.search(r"MARKET SKILLS for .*?:\n(\[.*\])", u, re.S).group(1))
        have_block = u.split("STUDENT PROJECTS")[0].lower()
        rows = []
        for m in market:
            first = re.split(r"[ /(]", m["skill"])[0].lower()
            st = "have" if first and first in have_block else "missing"
            rows.append({"skill": m["skill"], "status": st, "covered_by": m["skill"].split(" /")[0] if st == "have" else None,
                         "reason": "covered" if st == "have" else "needed for the role"})
        return {"rows": rows, "advice": "Start with the highest-importance missing skills."}

    def _roadmap(self, s, u):
        gaps = json.loads(re.search(r"GAPS \(ordered by priority\):\n(\[.*?\])\n", u, re.S).group(1))
        hours = int(re.search(r"Hours per week: (\d+)", u).group(1))
        return {"weeks": [{"week": i + 1, "focus": g["skill"], "skills": [g["skill"]], "topics": [f"Basics of {g['skill']}", "Practice"],
                           "resources": ["Official docs"], "deliverable": f"Notes + mini task on {g['skill']}", "hours": hours}
                          for i, g in enumerate(gaps[:8])], "note": "Mock plan"}

    def _project(self, s, u):
        gaps = json.loads(re.search(r"GAP skills: (\[.*?\])", u).group(1).replace("'", '"'))
        return {"projects": [{"title": f"Project {i+1}", "description": "Demo", "skills_covered": gaps[i:i + 3],
                              "difficulty": "easy", "duration_weeks": 2, "steps": ["Plan", "Build", "Deploy", "Write README"],
                              "resume_bullet": "Built X improving Y by [X%]"} for i in range(3)]}

    def _interview(self, s, u):
        if "STUDENT ANSWER" in u:
            score = 7 if len(u.split("STUDENT ANSWER:")[1]) > 80 else 3
            return {"score": score, "verdict": "Decent" if score > 5 else "Too short", "covered": ["definition"],
                    "missing": ["example"], "mistakes": [], "better_answer": ["Define", "Give example", "Trade-offs"]}
        return {"questions": [{"type": t, "skill": "SQL", "question": f"Mock {t} question {i}", "must_cover": ["a", "b", "c"]}
                              for i, t in enumerate(["technical"] * 4 + ["project", "behavioural"])]}

    def _progress(self, s, u):
        weeks = json.loads(re.search(r"COMPLETED WEEKS: (\[.*?\])\n", u, re.S).group(1))
        acquired = [sk for w in weeks if w["confidence_1to5"] >= 3 for sk in w["skills"]]
        return {"summary": f"Finished {len(weeks)} week(s).", "acquired_skills": acquired, "still_weak": [],
                "replan": bool(acquired), "reason": "skills acquired", "next_focus": "next gap", "pace": "keep going"}
