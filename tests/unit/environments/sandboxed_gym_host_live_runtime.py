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

"""Minimal Gym host runtime used by live OpenSandbox host / actor tests.

Serves ``GET /health`` and a stub ``POST /rollouts/run`` on port 8080 until the
production ``nmp-rl-training`` image is available.

The rollout stub returns NeMo-Gym-shaped results with synthetic token ids so
``SandboxedGymActor`` postprocess can run end-to-end without a full Gym tree.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


def _gym_result_for_example(example: dict, index: int) -> dict:
    # Contiguous prompt/generation token ids satisfy NemoGym postprocess.
    prompt_token_ids = list(range(10, 10 + 8 + index))
    generation_token_ids = [100 + index, 101 + index, 102 + index]
    generation_log_probs = [-0.1, -0.2, -0.3]
    return {
        "agent_ref": example.get("agent_ref")
        or {"name": "example_multi_step_simple_agent", "type": "responses_api_agents"},
        "responses_create_params": example.get("responses_create_params")
        or {"input": [{"role": "user", "content": "hi"}]},
        "response": {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"stub-{index}"}],
                    "prompt_token_ids": prompt_token_ids,
                    "generation_token_ids": generation_token_ids,
                    "generation_log_probs": generation_log_probs,
                }
            ]
        },
        "reward": 0.0,
        "job_id": os.environ.get("NMP_JOB_ID", ""),
        "environment_path": os.environ.get("NMP_ENVIRONMENT_PATH", ""),
        "work_path": os.environ.get("NMP_WORK_PATH", ""),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/health"):
            body = b'{"status":"ready"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/rollouts/run"):
            self.send_response(405)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/rollouts/run"):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                request = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                request = {}
            examples = request.get("examples") or []
            results = [
                _gym_result_for_example(example, i)
                for i, example in enumerate(examples)
            ]
            body = json.dumps(
                {
                    "results": results,
                    "job_id": os.environ.get("NMP_JOB_ID", ""),
                    "environment_path": os.environ.get("NMP_ENVIRONMENT_PATH", ""),
                    "work_path": os.environ.get("NMP_WORK_PATH", ""),
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def main() -> None:
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


if __name__ == "__main__":
    main()
