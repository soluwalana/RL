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

import pytest

from nemo_rl.utils.config_redact import (
    is_sensitive_config_key,
    normalize_config_key,
    redact_config_secrets,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api_key", "api_key"),
        ("API-Key", "api_key"),
        ("OpenAI.API.Key", "openai_api_key"),
        ("apiKey", "apikey"),
        ("  Token  ", "token"),
        ("---", ""),
    ],
)
def test_normalize_config_key(raw: str, expected: str) -> None:
    assert normalize_config_key(raw) == expected


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "API-Key",
        "apiKey",
        "apikey",
        "opensandbox_api_key",
        "OpenAI_API_Key",
        "password",
        "my_password",
        "secret",
        "token",
        "broker_token",
    ],
)
def test_is_sensitive_config_key_matches(key: str) -> None:
    assert is_sensitive_config_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "model_name",
        "tokenizer",
        "learning_rate",
        "",
        "---",
    ],
)
def test_is_sensitive_config_key_ignores_non_secrets(key: str) -> None:
    assert not is_sensitive_config_key(key)


def test_redact_config_secrets_nested() -> None:
    config = {
        "api_key": "x",
        "API-Key": "x",
        "apiKey": "x",
        "apikey": "x",
        "opensandbox_api_key": "x",
        "OpenAI_API_Key": "x",
        "broker_token": "x",
        "my_password": "x",
        "token": "x",
        "secret": "x",
        "model_name": "keep",
        "nested": {"api-key": "x", "lr": 1e-5},
        "items": [{"password": "x"}, "plain"],
    }

    redacted = redact_config_secrets(config)

    assert redacted["api_key"] == "<redacted>"
    assert redacted["API-Key"] == "<redacted>"
    assert redacted["apiKey"] == "<redacted>"
    assert redacted["apikey"] == "<redacted>"
    assert redacted["opensandbox_api_key"] == "<redacted>"
    assert redacted["OpenAI_API_Key"] == "<redacted>"
    assert redacted["broker_token"] == "<redacted>"
    assert redacted["my_password"] == "<redacted>"
    assert redacted["token"] == "<redacted>"
    assert redacted["secret"] == "<redacted>"
    assert redacted["model_name"] == "keep"
    assert redacted["nested"] == {"api-key": "<redacted>", "lr": 1e-5}
    assert redacted["items"] == [{"password": "<redacted>"}, "plain"]
