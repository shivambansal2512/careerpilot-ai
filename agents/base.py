"""Base class shared by all CareerPilot agents.

Every agent reads from and writes to one shared `state` dict (the student's case file)
and appends a record of what it did to state["trace"], so the UI can show exactly
which agent ran, which tool it used, how many LLM calls it made and how long it took.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager


class Agent:
    name = "Agent"
    goal = ""

    def __init__(self, llm):
        self.llm = llm

    @contextmanager
    def step(self, state: dict, action: str, tool: str | None = None):
        """Wrap one unit of agent work and log it to the trace."""
        t0 = time.time()
        calls0 = getattr(self.llm, "calls", 0)
        entry = {"agent": self.name, "action": action, "tool": tool or "—", "detail": "", "status": "running"}
        state.setdefault("trace", []).append(entry)
        try:
            yield entry
            entry["status"] = "done"
        except Exception as e:
            entry["status"] = "failed"
            entry["detail"] = (entry["detail"] + " " if entry["detail"] else "") + f"Error: {e}"
            raise
        finally:
            entry["seconds"] = round(time.time() - t0, 1)
            entry["llm_calls"] = getattr(self.llm, "calls", 0) - calls0

    @staticmethod
    def dump(obj, limit: int = 6000) -> str:
        s = json.dumps(obj, ensure_ascii=False, indent=1)
        return s if len(s) <= limit else s[:limit] + "\n...(truncated)"

    def system(self, instructions: str) -> str:
        return (
            f"AGENT_ID: {self.name}\n"
            f"You are the {self.name} inside CareerPilot AI, a multi-agent career mentor for "
            f"engineering students in India. Your goal: {self.goal}\n"
            "Be specific and realistic for entry-level hiring in India. Never invent URLs, "
            "salaries, company names or statistics.\n\n" + instructions
        )
