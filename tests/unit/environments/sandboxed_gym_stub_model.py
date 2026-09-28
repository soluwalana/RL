# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A stand-in vLLM endpoint, so a sandboxed rollout can run without a GPU.

Speaks the two routes ``nemo_gym``\'s ``VLLMModel`` calls upstream, not the Responses API the
*agent* sees. ``VLLMModel.responses`` converts Responses -> Chat Completions, calls the backend,
and converts back, so the backend must answer:

* ``POST {base_url}/chat/completions`` -- token ids ride in vLLM\'s logprobs shape, one entry per
  generated token with ``token`` spelled ``"token_id:<id>"``. ``VLLMModel`` reads
  ``generation_token_ids`` and ``generation_log_probs`` out of that block, and raises outright if
  it is missing, because the config sets ``return_token_id_information``.
* ``POST {base_url without /v1}/tokenize`` -- returns ``{"tokens": [...]}``, which becomes
  ``prompt_token_ids``. vLLM puts this one outside ``/v1``.

A hosted OpenAI-compatible endpoint cannot stand in: none of this is in the OpenAI contract.

Token ids are derived from the request rather than random, so a caller can assert that a given
prompt produced a given result instead of only that some result came back.

Run standalone::

    python sandboxed_gym_stub_model.py 8009
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Keeps generated ids clear of the prompt range so a mixed-up slice is visible in a diff.
GENERATION_TOKEN_BASE = 5000


def _message_text(message: object) -> str:
    """Flatten one chat message's content, which may be a string or a list of parts."""
    if isinstance(message, str):
        return message
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def _tokens_for(text: str) -> list[int]:
    """A stable, reversible-enough fake tokenizer: one id per character."""
    return [ord(c) % 256 for c in text] or [1]


def tokenize_for(body: dict) -> dict:
    """Answer vLLM's ``/tokenize``, which supplies ``prompt_token_ids``."""
    prompt = " ".join(
        _message_text(message) for message in body.get("messages") or []
    ).strip()
    return {"tokens": _tokens_for(prompt), "count": len(_tokens_for(prompt))}


def chat_completion_for(body: dict) -> dict:
    """Answer ``/chat/completions`` with per-token logprobs in vLLM's spelling."""
    prompt = " ".join(
        _message_text(message) for message in body.get("messages") or []
    ).strip()
    text = f"stub reply to {prompt}"
    token_ids = [GENERATION_TOKEN_BASE + (ord(c) % 256) for c in text]
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model") or "stub-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                # `token_id:` prefix and one entry per generated token: VLLMModel strips the
                # prefix to recover ids, and zips these against the generation one-to-one.
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token_id}", "logprob": -0.1}
                        for token_id in token_ids
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": len(_tokens_for(prompt)),
            "completion_tokens": len(token_ids),
            "total_tokens": len(_tokens_for(prompt)) + len(token_ids),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "stub-model"}]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        if path.endswith("/chat/completions"):
            self._send(200, chat_completion_for(body))
        elif path.endswith("/tokenize"):
            self._send(200, tokenize_for(body))
        else:
            self._send(404, {"error": f"unexpected path {self.path}"})

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler's signature
        pass


def serve(port: int) -> ThreadingHTTPServer:
    """Start the stub on ``port`` in a daemon thread and return the server."""
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8009)
    threading.Event().wait()
