from __future__ import annotations

from .base import Agent


class RoadmapAgent(Agent):
    name = "Roadmap Agent"
    goal = "turn the skill gaps into a realistic week-by-week learning plan that fits the student's weekly hours."

    def run(self, state: dict) -> dict:
        inp, gap = state["inputs"], state["gap"]
        hours = int(inp.get("hours_per_week") or 10)
        done = state.get("progress", {}).get("completed_skills", [])
        todo = [g for g in gap["gaps"] if g["skill"] not in done]
        with self.step(state, f"Plan weekly roadmap at {hours} h/week") as log:
            if not todo:
                state["roadmap"] = {"weeks": [], "note": "No open gaps. Spend time on projects and interview practice."}
                log["detail"] = "no gaps left"
                return state["roadmap"]
            system = self.system(
                "Build a week-by-week plan. Rules:\n"
                f"- Each week must fit in about {hours} hours (never more than {hours + 2}).\n"
                "- High-priority and 'missing' skills first; 'partial' skills get shorter revision weeks.\n"
                "- Group related skills in one week when it makes sense (e.g. SQL + DBMS).\n"
                "- Every week ends with a concrete deliverable the student can show (a small repo, notes, a solved set).\n"
                "- Resources: name well-known FREE resources only (official docs, freeCodeCamp, NPTEL, CS50, "
                "GeeksforGeeks, Striver's sheet, Kaggle Learn, etc.). Do not write URLs.\n"
                "- Between 3 and 12 weeks.\n"
                "JSON schema: {\"weeks\": [{\"week\": int, \"focus\": str, \"skills\": [str], \"topics\": [str], "
                "\"resources\": [str], \"deliverable\": str, \"hours\": int}], \"note\": str}"
            )
            user = (f"Target role: {inp['target_role']}. Year: {inp.get('year')}. Hours per week: {hours}.\n"
                    f"Already completed through earlier roadmap: {done or 'nothing yet'}.\n"
                    f"GAPS (ordered by priority):\n"
                    f"{self.dump([{k: g[k] for k in ('skill', 'status', 'importance', 'priority')} for g in todo])}\n"
                    f"Student strengths to build on: {state['profile'].get('strengths', [])}")
            data = self.llm.json(system, user, required=("weeks",))
            weeks = []
            for i, w in enumerate(data.get("weeks") or []):
                try:
                    h = int(w.get("hours", hours))
                except (TypeError, ValueError):
                    h = hours
                weeks.append({"week": i + 1, "focus": w.get("focus", ""), "skills": w.get("skills") or [],
                              "topics": w.get("topics") or [], "resources": w.get("resources") or [],
                              "deliverable": w.get("deliverable", ""), "hours": min(h, hours + 2)})
            roadmap = {"weeks": weeks[:12], "note": data.get("note", ""), "hours_per_week": hours}
            state["roadmap"] = roadmap
            covered = {s.lower() for w in weeks for s in w["skills"]}
            missed = [g["skill"] for g in todo if g["priority"] == "high" and g["skill"].lower() not in covered]
            log["detail"] = f"{len(weeks)} weeks" + (f" · not explicitly covered: {', '.join(missed)}" if missed else "")
        return roadmap
