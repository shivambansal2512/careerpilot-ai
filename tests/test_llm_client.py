"""Tests the real LLM client against a local fake OpenAI-compatible server."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import llm as llm_mod  # noqa: E402

SEEN = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body["model"])
        if body["model"] == "retired-model":
            self.send_response(404); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":{"message":"model not found","code":"model_not_found"}}'); return
        if self.headers.get("Authorization") == "Bearer bad":
            self.send_response(401); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error":{"message":"invalid API key"}}'); return
        content = '```json\n{"skills": [{"name": "SQL", "level": "beginner"}]}\n```'
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(resp).encode())


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    llm_mod.PROVIDERS["fake"] = {"base_url": f"http://127.0.0.1:{srv.server_port}/v1", "label": "Fake",
                                 "key_name": "FAKE_KEY", "models": ["retired-model", "good-model"]}
    yield
    srv.shutdown()


def test_fallback_and_json(server):
    c = llm_mod.LLM(provider="fake", api_key="ok")
    out = c.json("sys", "user", required=("skills",))
    assert out["skills"][0]["name"] == "SQL"
    assert c.model == "good-model" and SEEN[:2] == ["retired-model", "good-model"]
    assert c.calls == 1 and c.total_tokens == 15


def test_bad_key_message(server):
    c = llm_mod.LLM(provider="fake", api_key="bad", model="good-model")
    with pytest.raises(llm_mod.LLMError, match="API key was rejected"):
        c.chat("s", "u")
