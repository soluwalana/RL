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

"""Redact credential-looking values from printable config dumps."""

from __future__ import annotations

import re
from typing import Any

# Matched as an exact key or as a key suffix after normalization.
SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
    }
)


def normalize_config_key(key: str) -> str:
    """Lowercase and map every non-alpha run to a single underscore."""
    return re.sub(r"[^a-z]+", "_", str(key).lower()).strip("_")


def is_sensitive_config_key(key: str) -> bool:
    normalized = normalize_config_key(key)
    if not normalized:
        return False
    # Direct match or underscore-delimited suffix
    # (e.g. opensandbox_api_key, OpenAI-API-Key → openai_api_key).
    # Require "_" before a suffix so "tokenizer" does not match "token".
    return any(
        normalized == token or normalized.endswith("_" + token)
        for token in SENSITIVE_CONFIG_KEYS
    )


def redact_config_secrets(value: Any) -> Any:
    """Return a printable config copy with credential values removed."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if is_sensitive_config_key(key):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_config_secrets(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_config_secrets(item) for item in value]
    return value
