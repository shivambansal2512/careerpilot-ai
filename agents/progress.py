from __future__ import annotations

from .base import Agent


class ProgressAgent(Agent):
    name = "Progress Agent"
    goal = "review what the student finished, decide which skills are now acquired, and decide whether the plan needs re-planning."

    def run(self, state: dict, completed_weeks: list[int], confidence: dict[int, int], notes: str = "") -> dict:
        weeks = {w["week"]: w for w in state.get("roadmap", {}).get("weeks", [])}
        finished = [weeks[w] for w in completed_weeks if w in weeks]
        if not finished:
            raise ValueError("Tick at least one completed week first.")
        scores = [s for s in state.get("interview", {}).get("attempts", {}).values()]
        with self.step(state, f"Review {len(finished)} completed week(s)") as log:
            system = self.system(
                "Review the student's progress. A skill counts as acquired only if its week is done AND the "
                "student's confidence is 3 or more out of 5 (or their interview answers on it scored 6+). "
                "Decide whether to re-plan: re-plan when skills were acquired, when confidence is low (<=2) "
                "on a week, or when the notes say the pace is wrong.\n"
                "JSON schema: {\"summary\": str, \"acquired_skills\": [str], \"still_weak\": [str], "
                "\"replan\": bool, \"reason\": str, \"next_focus\": str, "
                "\"pace\": \"slow down|keep going|speed up\"}"
            )
            user = (f"COMPLETED WEEKS: {self.dump([{'week': w['week'], 'skills': w['skills'], 'confidence_1to5': confidence.get(w['week'], 3)} for w in finished])}\n"
                    f"STUDENT NOTES: {notes or '(none)'}\n"
                    f"INTERVIEW ATTEMPTS: {self.dump([[a['eval'].get('score') for a in s] for s in scores], 800)}\n"
                    f"OPEN GAPS BEFORE: {[g['skill'] for g in state['gap']['gaps']]}")
            data = self.llm.json(system, user, required=("acquired_skills", "replan"))
            gap_names = {g["skill"].lower(): g["skill"] for g in state["gap"]["gaps"]}
            acquired = [gap_names[s.lower()] for s in data.get("acquired_skills") or [] if s.lower() in gap_names]

            prog = state.setdefault("progress", {"completed_weeks": [], "completed_skills": [], "reviews": []})
            prog["completed_weeks"] = sorted(set(prog["completed_weeks"]) | set(completed_weeks))
            prog["completed_skills"] = sorted(set(prog["completed_skills"]) | set(acquired))
            data["acquired_skills"] = acquired
            prog["reviews"].append(data)

            # write newly acquired skills back into the profile so the Skill-Gap Agent sees them
            have = {s["name"].lower() for s in state["profile"]["skills"]}
            for s in acquired:
                if s.lower() not in have:
                    confs = [confidence.get(w["week"], 3) for w in finished
                             if s.lower() in {x.lower() for x in w["skills"]}]
                    lvl = "intermediate" if (min(confs) if confs else 3) >= 4 else "beginner"
                    state["profile"]["skills"].append({"name": s, "level": lvl, "evidence": "Completed CareerPilot roadmap week"})
            log["detail"] = f"acquired: {', '.join(acquired) or 'none'} · replan: {bool(data.get('replan'))}"
        return data
