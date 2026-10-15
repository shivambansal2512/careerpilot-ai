"""Orchestrator: decides which agent runs next and keeps the shared state.

Flow for a new analysis:
    Profile Analyzer -> Job-Market Agent -> Skill-Gap Agent
        -> (readiness >= 85?) skip roadmap, go to interview-first mode
        -> Roadmap Agent -> Project Agent -> Interview Coach

Flow when the student reports progress (closed feedback loop):
    Progress Agent -> (replan?) Skill-Gap Agent -> Roadmap Agent
"""
from __future__ import annotations

from typing import Callable

from agents import (InterviewCoachAgent, JobMarketAgent, ProfileAnalyzerAgent, ProgressAgent,
                    ProjectAdvisorAgent, RoadmapAgent, SkillGapAgent)

INTERVIEW_FIRST_THRESHOLD = 85


class Orchestrator:
    def __init__(self, llm, adzuna_id=None, adzuna_key=None):
        self.llm = llm
        self.profile = ProfileAnalyzerAgent(llm)
        self.market = JobMarketAgent(llm, adzuna_id, adzuna_key)
        self.gap = SkillGapAgent(llm)
        self.roadmap = RoadmapAgent(llm)
        self.projects = ProjectAdvisorAgent(llm)
        self.coach = InterviewCoachAgent(llm)
        self.progress = ProgressAgent(llm)

    @staticmethod
    def new_state(inputs: dict, keep: dict | None = None) -> dict:
        state = {"inputs": inputs, "trace": [], "decisions": []}
        if keep and "market_cache" in keep:
            state["market_cache"] = keep["market_cache"]
        return state

    def _decide(self, state, text):
        state["decisions"].append(text)
        state["trace"].append({"agent": "Orchestrator", "action": "Route", "tool": "—", "detail": text,
                               "status": "done", "seconds": 0, "llm_calls": 0})

    def run_analysis(self, state: dict, on_step: Callable[[str], None] = lambda s: None) -> dict:
        on_step(self.profile.name)
        self.profile.run(state)
        on_step(self.market.name)
        self.market.run(state)
        on_step(self.gap.name)
        gap = self.gap.run(state)

        if gap["readiness"] >= INTERVIEW_FIRST_THRESHOLD:
            self._decide(state, f"Readiness {gap['readiness']} ≥ {INTERVIEW_FIRST_THRESHOLD}: skip the learning "
                                "roadmap and go straight to projects and interview practice.")
            state["roadmap"] = {"weeks": [], "note": "You already cover most requirements. Focus on projects and interviews."}
        else:
            self._decide(state, f"Readiness {gap['readiness']} < {INTERVIEW_FIRST_THRESHOLD}: build a learning roadmap "
                                f"for {len(gap['gaps'])} open gaps.")
            on_step(self.roadmap.name)
            self.roadmap.run(state)

        on_step(self.projects.name)
        self.projects.run(state)
        on_step(self.coach.name)
        self.coach.run(state)
        return state

    def answer_question(self, state: dict, qid: str, answer: str) -> dict:
        return self.coach.evaluate(state, qid, answer)

    def report_progress(self, state: dict, completed_weeks, confidence, notes="",
                        on_step: Callable[[str], None] = lambda s: None) -> dict:
        on_step(self.progress.name)
        review = self.progress.run(state, completed_weeks, confidence, notes)
        if review.get("replan"):
            self._decide(state, f"Progress Agent asked to re-plan ({review.get('reason', '')}). "
                                "Re-running Skill-Gap and Roadmap agents.")
            old_weeks = state.get("roadmap", {}).get("weeks", [])
            on_step(self.gap.name)
            self.gap.run(state)
            on_step(self.roadmap.name)
            self.roadmap.run(state)
            state.setdefault("progress", {})["previous_roadmaps"] = (
                state["progress"].get("previous_roadmaps", []) + [old_weeks])
            # completed weeks refer to the old plan; reset ticks for the new plan
            state["progress"]["completed_weeks"] = []
        else:
            self._decide(state, "Progress Agent: no re-plan needed, keep following the current roadmap.")
        return review
