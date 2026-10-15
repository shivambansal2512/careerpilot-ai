from __future__ import annotations

from .base import Agent

LEVELS = {"beginner", "intermediate", "advanced"}


class ProfileAnalyzerAgent(Agent):
    name = "Profile Analyzer"
    goal = "turn a resume and/or a free-text skills list into a structured student profile with evidence-based skill levels."

    def run(self, state: dict) -> dict:
        inp = state["inputs"]
        source = []
        if inp.get("resume_text"):
            source.append("RESUME TEXT:\n" + inp["resume_text"])
        if inp.get("skills_text"):
            source.append("SKILLS THE STUDENT TYPED:\n" + inp["skills_text"])
        if not source:
            raise ValueError("Upload a resume or type your skills first.")

        with self.step(state, "Read resume / skills and build profile",
                       tool="resume_parser" if inp.get("resume_text") else None) as log:
            system = self.system(
                "Extract the student's profile. For each skill estimate a level from the evidence:\n"
                "- beginner: only listed, or coursework\n"
                "- intermediate: used in a project or internship\n"
                "- advanced: multiple substantial projects, internship ownership, or strong competitive record\n"
                "Normalise names (e.g. 'reactjs' -> 'React', 'DS&A' -> 'DSA'). Do not add skills that have no evidence.\n"
                "JSON schema: {\"name\": str|null, \"education\": str|null, \"year\": str|null, "
                "\"summary\": str (2 sentences), "
                "\"skills\": [{\"name\": str, \"level\": \"beginner|intermediate|advanced\", \"evidence\": str}], "
                "\"projects\": [{\"name\": str, \"tech\": [str], \"what\": str}], "
                "\"strengths\": [str]}"
            )
            user = "\n\n".join(source) + f"\n\nTarget role: {inp.get('target_role')}. Year: {inp.get('year')}."
            prof = self.llm.json(system, user, required=("skills",))

            skills, seen = [], set()
            for s in prof.get("skills") or []:
                n = str(s.get("name", "")).strip()
                if not n or n.lower() in seen:
                    continue
                seen.add(n.lower())
                lvl = str(s.get("level", "beginner")).lower()
                skills.append({"name": n, "level": lvl if lvl in LEVELS else "beginner",
                               "evidence": str(s.get("evidence", ""))[:200]})
            prof["skills"] = skills
            prof["projects"] = prof.get("projects") or []
            prof["strengths"] = prof.get("strengths") or []
            state["profile"] = prof
            log["detail"] = f"{len(skills)} skills, {len(prof['projects'])} projects found"
        return prof
