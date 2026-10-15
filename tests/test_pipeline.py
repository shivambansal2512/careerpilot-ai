"""Run with:  python -m pytest -q   (from the project folder). No API key needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import LLMError, extract_json  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from tests.mock_llm import MockLLM  # noqa: E402
from tools.job_search import skill_frequency  # noqa: E402

INPUTS = {"resume_text": "", "skills_text": "Java, DSA, SQL, Git, React, Node.js",
          "target_role": "Software Engineer (Backend)", "year": "3rd year", "hours_per_week": 10, "location": ""}


def run():
    llm = MockLLM()
    orch = Orchestrator(llm)
    state = Orchestrator.new_state(dict(INPUTS))
    steps = []
    orch.run_analysis(state, steps.append)
    return orch, state, steps, llm


def test_full_pipeline_runs_every_agent():
    orch, state, steps, llm = run()
    assert steps == ["Profile Analyzer", "Job-Market Agent", "Skill-Gap Agent",
                     "Roadmap Agent + Project Agent + Interview Coach (in parallel)"]
    ran = {t["agent"] for t in state["trace"]}
    assert {"Roadmap Agent", "Project Agent", "Interview Coach"} <= ran
    assert llm.calls == 6
    assert 0 <= state["gap"]["readiness"] <= 100
    assert state["roadmap"]["weeks"] and len(state["projects"]["projects"]) == 3
    assert len(state["interview"]["questions"]) == 6
    assert all(t["status"] == "done" for t in state["trace"])


def test_gap_uses_market_baseline_and_scores():
    _, state, _, _ = run()
    names = [r["skill"] for r in state["gap"]["rows"]]
    assert "DSA" in names and "SQL" in names
    have = {r["skill"] for r in state["gap"]["rows"] if r["status"] == "have"}
    assert {"DSA", "SQL", "Git"} <= have
    assert all(g["status"] != "have" for g in state["gap"]["gaps"])


def test_interview_feedback():
    orch, state, _, _ = run()
    ev = orch.answer_question(state, "q1", "An index is a data structure that speeds up lookups, e.g. a B-tree on a column, "
                                           "at the cost of slower writes and extra storage.")
    assert 0 <= ev["score"] <= 10 and ev["better_answer"]
    with pytest.raises(ValueError):
        orch.answer_question(state, "q1", "idk")


def test_progress_triggers_replan_and_raises_readiness():
    orch, state, _, _ = run()
    before = state["gap"]["readiness"]
    first_weeks = state["roadmap"]["weeks"][:2]
    review = orch.report_progress(state, [w["week"] for w in first_weeks], {w["week"]: 4 for w in first_weeks})
    assert review["replan"] is True
    assert set(review["acquired_skills"]) == {s for w in first_weeks for s in w["skills"]}
    assert state["gap"]["readiness"] > before          # gap agent re-ran on the updated profile
    assert len(state["readiness_history"]) == 2
    assert any("re-plan" in d for d in state["decisions"])


def test_high_readiness_skips_roadmap():
    llm = MockLLM()
    orch = Orchestrator(llm)
    inp = dict(INPUTS, skills_text="DSA SQL REST Git Java OOP DBMS Spring Operating Docker Computer System Cloud Unit (advanced)")
    state = Orchestrator.new_state(inp)
    steps = []
    orch.run_analysis(state, steps.append)
    assert state["gap"]["readiness"] >= 85
    assert not any("Roadmap" in s for s in steps) and state["roadmap"]["weeks"] == []


def test_invalid_json_is_retried():
    llm = MockLLM(bad_json_once=True)
    orch = Orchestrator(llm)
    state = Orchestrator.new_state(dict(INPUTS))
    orch.profile.run(state)
    assert llm.calls == 2 and state["profile"]["skills"]


def test_no_input_is_rejected():
    orch = Orchestrator(MockLLM())
    with pytest.raises(ValueError):
        orch.profile.run(Orchestrator.new_state(dict(INPUTS, skills_text="")))


def test_market_cache_reused():
    llm = MockLLM()
    orch = Orchestrator(llm)
    s1 = Orchestrator.new_state(dict(INPUTS))
    orch.run_analysis(s1)
    calls = llm.calls
    s2 = Orchestrator.new_state(dict(INPUTS), keep=s1)
    orch.run_analysis(s2)
    assert llm.calls - calls == 5   # job-market agent skipped thanks to cache


def test_extract_json_and_frequency():
    assert extract_json('text ```json\n{"a": 1}\n``` more') == {"a": 1}
    posts = [{"title": "Backend Dev", "description": "Java, Spring Boot, SQL"},
             {"title": "SDE", "description": "Python, SQL, Docker"}]
    f = skill_frequency(["SQL", "Docker", "Java"], posts, {"Java": ["spring boot"]})
    assert f == {"SQL": 100, "Docker": 50, "Java": 50}
