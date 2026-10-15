from __future__ import annotations

from .base import Agent


class InterviewCoachAgent(Agent):
    name = "Interview Coach"
    goal = "prepare the student for real interviews: ask role-specific questions and grade their answers honestly."

    def run(self, state: dict) -> dict:
        inp, gap, prof = state["inputs"], state["gap"], state["profile"]
        with self.step(state, "Generate personalised interview questions") as log:
            system = self.system(
                "Write 6 interview questions for this student:\n"
                "- 3 technical questions on their highest-priority GAP or partial skills,\n"
                "- 1 technical question on a skill they already have (go one level deeper than their level),\n"
                "- 1 question about one of their own projects (use the real project name if available),\n"
                "- 1 behavioural question (HR round, STAR format).\n"
                "For each, list 3-5 points a strong answer must cover.\n"
                "JSON schema: {\"questions\": [{\"id\": str, \"type\": \"technical|project|behavioural\", "
                "\"skill\": str, \"question\": str, \"must_cover\": [str]}]}"
            )
            user = (f"Target role: {inp['target_role']}\n"
                    f"Gaps: {[(g['skill'], g['status']) for g in gap['gaps'][:6]]}\n"
                    f"Skills: {[(s['name'], s['level']) for s in prof['skills']]}\n"
                    f"Projects: {self.dump(prof.get('projects', []), 1500)}")
            data = self.llm.json(system, user, required=("questions",))
            qs = []
            for i, q in enumerate((data.get("questions") or [])[:8]):
                qs.append({"id": f"q{i+1}", "type": q.get("type", "technical"), "skill": q.get("skill", ""),
                           "question": q.get("question", ""), "must_cover": q.get("must_cover") or []})
            state["interview"] = {"questions": qs, "attempts": state.get("interview", {}).get("attempts", {})}
            log["detail"] = f"{len(qs)} questions"
        return state["interview"]

    def evaluate(self, state: dict, qid: str, answer: str) -> dict:
        q = next((q for q in state["interview"]["questions"] if q["id"] == qid), None)
        if q is None:
            raise ValueError("Unknown question.")
        if len(answer.strip()) < 15:
            raise ValueError("Write a fuller answer (at least a couple of sentences) before asking for feedback.")
        with self.step(state, f"Grade answer to {qid}") as log:
            system = self.system(
                "Grade the student's interview answer like a fair but strict interviewer at an Indian product company.\n"
                "score: 0-10 (6 = acceptable at entry level, 8+ = strong). Base it on the must-cover points, "
                "correctness and clarity. Do not inflate.\n"
                "JSON schema: {\"score\": int, \"verdict\": str, \"covered\": [str], \"missing\": [str], "
                "\"mistakes\": [str], \"better_answer\": [str] (4-6 bullet points of a model answer)}"
            )
            user = (f"QUESTION ({q['type']}, {q['skill']}): {q['question']}\n"
                    f"MUST COVER: {q['must_cover']}\n\nSTUDENT ANSWER:\n{answer.strip()[:3000]}")
            ev = self.llm.json(system, user, temperature=0.2, required=("score",))
            try:
                ev["score"] = max(0, min(10, int(ev.get("score", 0))))
            except (TypeError, ValueError):
                ev["score"] = 0
            state["interview"].setdefault("attempts", {}).setdefault(qid, []).append({"answer": answer, "eval": ev})
            log["detail"] = f"score {ev['score']}/10"
        return ev
