from __future__ import annotations

from .base import Agent


class ProjectAdvisorAgent(Agent):
    name = "Project Agent"
    goal = "recommend portfolio projects that close the student's biggest skill gaps and look credible to recruiters."

    def run(self, state: dict) -> dict:
        inp, gap, prof = state["inputs"], state["gap"], state["profile"]
        gap_names = [g["skill"] for g in gap["gaps"]]
        with self.step(state, "Recommend gap-closing projects") as log:
            system = self.system(
                "Suggest exactly 3 portfolio projects for this student, easiest first. Each must:\n"
                "- use several of the student's GAP skills (name them exactly as given in the gap list),\n"
                "- be different from the projects already on their resume,\n"
                "- be finishable in 1-3 weeks by one student,\n"
                "- include 4-6 build steps and one resume bullet written with a measurable result placeholder like [X%].\n"
                "JSON schema: {\"projects\": [{\"title\": str, \"description\": str, \"skills_covered\": [str], "
                "\"difficulty\": \"easy|medium|hard\", \"duration_weeks\": int, \"steps\": [str], \"resume_bullet\": str}]}"
            )
            user = (f"Target role: {inp['target_role']}\nGAP skills: {gap_names}\n"
                    f"Skills they already have: {[s['name'] for s in prof['skills']]}\n"
                    f"Existing projects: {[p.get('name') for p in prof.get('projects', [])]}")
            data = self.llm.json(system, user, required=("projects",))
            lower_gaps = {g.lower(): g for g in gap_names}
            projects = []
            for p in (data.get("projects") or [])[:3]:
                covered = [str(s) for s in p.get("skills_covered") or []]
                closes = sorted({lower_gaps[c.lower()] for c in covered if c.lower() in lower_gaps})
                projects.append({**p, "skills_covered": covered, "gaps_closed": closes})
            state["projects"] = {"projects": projects}
            log["detail"] = "; ".join(f"{p.get('title')} (closes {len(p['gaps_closed'])} gaps)" for p in projects)
        return state["projects"]
