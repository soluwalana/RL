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

"""Tests for the environment-package helpers shared by both Gym integration modes.

Colocated (``nemo_rl.environments.nemo_gym``) and sandboxed
(``nemo_rl.environments.sandbox.gym_host_runtime``) call the same functions, so the
behaviour is pinned once here and each caller's own module tests only cover its wiring.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.environments import gym_env_package as pkg


def _ok():
    """Stand-in for a successful subprocess.run result."""
    return SimpleNamespace(returncode=0, stderr="")


def _make_server_venv(venv_root, server_type, name):
    """Mirror Gym's <uv_venv_dir>/<server_type>/<name>/.venv layout."""
    python = venv_root / server_type / name / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    return python


def _make_package(root, wheels=("ascii_tree-0.1.0-py3-none-any.whl",)):
    wheels_dir = root / "wheels"
    wheels_dir.mkdir(parents=True)
    for name in wheels:
        (wheels_dir / name).touch()
    return root


# --------------------------------------------------------------------------------------
# register_environment_search_root
# --------------------------------------------------------------------------------------


def test_register_environment_search_root_prepends_the_package(tmp_path, monkeypatch):
    """native-v1 server dirs are resolved by NAME, so the package must be a search root.

    Prepended, not appended: earlier roots win on a name collision, which is what lets a
    user's environment shadow a built-in of the same name.
    """
    env_root = tmp_path / "environment"
    env_root.mkdir()
    monkeypatch.setenv(pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR, "/opt/plugins")

    result = pkg.register_environment_search_root(str(env_root))

    assert result.split(os.pathsep) == [str(env_root), "/opt/plugins"]
    assert os.environ[pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR] == result


def test_register_environment_search_root_is_idempotent(tmp_path, monkeypatch):
    """Re-registering must not grow the path; a duplicated root is still one root."""
    env_root = tmp_path / "environment"
    env_root.mkdir()
    monkeypatch.delenv(pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR, raising=False)

    pkg.register_environment_search_root(str(env_root))
    assert pkg.register_environment_search_root(str(env_root)) == str(env_root)


@pytest.mark.parametrize("env_root", [None, "", "   "])
def test_register_environment_search_root_noop_without_a_package(env_root, monkeypatch):
    """Standalone NeMo-RL and bundled config_paths runs have no package to register."""
    monkeypatch.delenv(pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR, raising=False)

    assert pkg.register_environment_search_root(env_root) is None
    assert pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR not in os.environ


def test_register_environment_search_root_raises_on_a_missing_directory(
    tmp_path, monkeypatch
):
    """A root that does not resolve is worse than an error: it trains the wrong environment.

    Gym falls through to its install root when an extra root does not contain the named
    server dir, so it would silently start a BUILT-IN server of the same name and the job
    would run to completion against an environment the user never uploaded.
    """
    monkeypatch.delenv(pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR, raising=False)

    with pytest.raises(FileNotFoundError, match="not a directory"):
        pkg.register_environment_search_root(str(tmp_path / "never-staged"))

    # And it must not have half-registered before failing.
    assert pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR not in os.environ


def test_register_environment_search_root_raises_when_the_path_is_a_file(
    tmp_path, monkeypatch
):
    """Same hazard as a missing dir: a file cannot hold a server tree."""
    not_a_dir = tmp_path / "environment"
    not_a_dir.write_text("", encoding="utf-8")
    monkeypatch.delenv(pkg.NEMO_GYM_EXTRA_ROOTS_ENV_VAR, raising=False)

    with pytest.raises(FileNotFoundError, match="not a directory"):
        pkg.register_environment_search_root(str(not_a_dir))


# --------------------------------------------------------------------------------------
# wheel_requirement
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("ascii_tree-0.1.0-py3-none-any.whl", "ascii_tree==0.1.0"),
        ("verifiers-0.1.14-py3-none-any.whl", "verifiers==0.1.14"),
        # Optional build tag sits between version and python tag.
        ("foo-1.2.3-1-cp313-cp313-manylinux_2_17_x86_64.whl", "foo==1.2.3"),
    ],
)
def test_wheel_requirement_parses_pep427_filenames(filename, expected):
    assert pkg.wheel_requirement(Path(filename)) == expected


def test_wheel_requirement_rejects_non_wheel_filenames():
    with pytest.raises(ValueError, match="PEP 427"):
        pkg.wheel_requirement(Path("not-a-wheel.whl"))


# --------------------------------------------------------------------------------------
# env_package_venv_pythons
# --------------------------------------------------------------------------------------


def test_env_package_venv_pythons_skips_incomplete_venvs(tmp_path):
    """A venv dir without an interpreter is half-built; installing into it would fail."""
    venv_root = tmp_path / "gym_venvs"
    good = _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")
    (venv_root / "responses_api_agents" / "half_built" / ".venv").mkdir(parents=True)

    assert pkg.env_package_venv_pythons(str(venv_root)) == [good]


def test_env_package_venv_pythons_excludes_model_servers(tmp_path):
    """Model servers proxy to vLLM and never import the environment package."""
    venv_root = tmp_path / "gym_venvs"
    agent = _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")
    _make_server_venv(venv_root, "responses_api_models", "vllm_model")

    assert pkg.env_package_venv_pythons(str(venv_root)) == [agent]


def test_env_package_venv_pythons_includes_resources_servers(tmp_path):
    """wheels-v1 declares no adapter, so its code may be a verifier, not an agent."""
    venv_root = tmp_path / "gym_venvs"
    agent = _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")
    verifier = _make_server_venv(venv_root, "resources_servers", "my_env")

    assert pkg.env_package_venv_pythons(str(venv_root)) == [agent, verifier]


def test_env_package_venv_pythons_empty_for_a_missing_root(tmp_path):
    assert pkg.env_package_venv_pythons(str(tmp_path / "nope")) == []


# --------------------------------------------------------------------------------------
# install_environment_wheels
# --------------------------------------------------------------------------------------


def test_install_environment_wheels_installs_offline_into_server_venvs(
    tmp_path, monkeypatch
):
    """The package's vendored closure must reach the server venv without an index.

    Gym builds that venv from the server's requirements.txt, which carries the framework
    only, so nothing else puts the environment package where ``vf.load_environment`` can
    import it.
    """
    env_root = _make_package(
        tmp_path / "environment",
        wheels=(
            "ascii_tree-0.1.0-py3-none-any.whl",
            "verifiers-0.1.14-py3-none-any.whl",
        ),
    )
    venv_root = tmp_path / "gym_venvs"
    python = _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")

    calls = []
    monkeypatch.setattr(
        pkg.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or _ok()
    )

    pkg.install_environment_wheels({pkg.UV_VENV_DIR_KEY: str(venv_root)}, str(env_root))

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    # stderr captured, not inherited: the failure path re-raises with the resolver's
    # explanation instead of a CalledProcessError that renders the whole argv.
    assert kwargs["stderr"] is pkg.subprocess.PIPE
    assert cmd[:3] == ["uv", "pip", "install"]
    assert str(python) in cmd
    # No index may be consulted: the sandbox cannot reach the one the env was published to.
    assert "--no-index" in cmd
    assert f"--find-links={env_root / 'wheels'}" in cmd
    # Names, not paths: `uv pip install <path>.whl` uninstalls and reinstalls even when that
    # exact version is already present, which would rebuild the venv on every spin-up.
    assert "ascii-tree" in cmd
    assert "verifiers" in cmd
    assert not any(c.endswith(".whl") for c in cmd)
    # Never pinned: a pin overwrites what the server already resolved and imported.
    assert not any("==" in c for c in cmd)


def test_install_environment_wheels_installs_into_every_env_code_venv(
    tmp_path, monkeypatch
):
    """Both server tiers that can host environment code get the closure."""
    env_root = _make_package(tmp_path / "environment")
    venv_root = tmp_path / "gym_venvs"
    agent = _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")
    verifier = _make_server_venv(venv_root, "resources_servers", "my_env")

    calls = []
    monkeypatch.setattr(
        pkg.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _ok()
    )

    pkg.install_environment_wheels({pkg.UV_VENV_DIR_KEY: str(venv_root)}, str(env_root))

    installed_into = [cmd[cmd.index("--python") + 1] for cmd in calls]
    assert installed_into == [str(agent), str(verifier)]


def test_install_environment_wheels_raises_when_no_server_venv_exists(
    tmp_path, monkeypatch
):
    """Wheels to install but nowhere to put them must fail here, not at the first rollout."""
    env_root = _make_package(tmp_path / "environment")
    monkeypatch.setattr(pkg.subprocess, "run", lambda *a, **k: _ok())

    with pytest.raises(RuntimeError, match=r"no\s+server\s+venv"):
        pkg.install_environment_wheels(
            {pkg.UV_VENV_DIR_KEY: str(tmp_path / "empty")}, str(env_root)
        )


def test_install_environment_wheels_raises_without_a_venv_root(tmp_path, monkeypatch):
    """No uv_venv_dir with wheels present is fatal, not a warning.

    Gym then puts venvs at <server_dir>/.venv, which is not derivable here, so the closure
    cannot be installed at all -- and the run dies at the first rollout with an ImportError
    that says nothing about the cause. Fail at the cause instead.
    """
    env_root = _make_package(tmp_path / "environment")

    def _never(*a, **k):
        raise AssertionError("nothing can be installed without a known venv root")

    monkeypatch.setattr(pkg.subprocess, "run", _never)

    with pytest.raises(RuntimeError, match="NEMO_GYM_VENV_DIR"):
        pkg.install_environment_wheels({}, str(env_root))


def test_install_environment_wheels_noop_without_wheels(tmp_path, monkeypatch):
    """native-v1 packages and bundled config_paths runs ship no wheels; both are valid."""
    env_root = tmp_path / "environment"
    (env_root / "configs").mkdir(parents=True)

    def _never(*a, **k):
        raise AssertionError("no install should run when the package ships no wheels")

    monkeypatch.setattr(pkg.subprocess, "run", _never)
    pkg.install_environment_wheels(
        {pkg.UV_VENV_DIR_KEY: str(tmp_path / "gym_venvs")}, str(env_root)
    )


def test_install_environment_wheels_noop_without_wheels_and_without_venv_root(tmp_path):
    """The venv-root check must stay behind the wheels check.

    A native-v1 run outside a container has neither, and raising on it would break a
    configuration that is correct.
    """
    env_root = tmp_path / "environment"
    env_root.mkdir()

    pkg.install_environment_wheels({}, str(env_root))


@pytest.mark.parametrize("env_root", [None, "", "   "])
def test_install_environment_wheels_noop_without_environment_path(
    env_root, monkeypatch
):
    """Standalone NeMo-RL sets no environment path and must stay unaffected."""

    def _never(*a, **k):
        raise AssertionError("no install should run outside a platform job")

    monkeypatch.setattr(pkg.subprocess, "run", _never)
    pkg.install_environment_wheels({}, env_root)


# --------------------------------------------------------------------------------------
# wheelhouse_requirements
# --------------------------------------------------------------------------------------


def test_wheelhouse_requirements_never_pins():
    """Pinning rewrites packages the server already resolved -- and already imported."""
    wheels = [
        Path("ascii_tree-0.1.5-py3-none-any.whl"),
        Path("verifiers-0.1.14-py3-none-any.whl"),
    ]
    assert pkg.wheelhouse_requirements(wheels) == ["ascii-tree", "verifiers"]


def test_wheelhouse_requirements_collapses_duplicated_distributions():
    """Two versions of one project must not become two contradictory pins.

    A wheelhouse is a candidate pool, not a lock file; pinning every file turns a
    perfectly installable pool into `you require xxhash==3.8.1 and xxhash==4.0.0, we can
    conclude that your requirements are unsatisfiable`.
    """
    wheels = [
        Path("ascii_tree-0.1.5-py3-none-any.whl"),
        Path("xxhash-3.8.1-cp313-cp313-manylinux_2_17_x86_64.whl"),
        Path("xxhash-4.0.0-cp313-cp313-manylinux_2_17_x86_64.whl"),
    ]

    assert pkg.wheelhouse_requirements(wheels) == ["ascii-tree", "xxhash"]

    # One requirement per project, never two pins for the same one.
    assert len(pkg.wheelhouse_requirements(wheels)) == 2


def test_wheelhouse_requirements_falls_back_to_the_normalized_name():
    """The unpinned fallback must be the PEP 503 name, which is what installers accept.

    Wheel filenames escape the distribution with ``_``; a requirement spelled that way
    still resolves, but the normalized form is the canonical one and is what groups the
    two files into one project in the first place.
    """
    wheels = [
        Path("charset_normalizer-3.4.9-py3-none-any.whl"),
        Path("charset_normalizer-3.5.0-py3-none-any.whl"),
    ]

    assert pkg.wheelhouse_requirements(wheels) == ["charset-normalizer"]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("charset_normalizer-3.5.0-py3-none-any.whl", "charset-normalizer"),
        ("ascii_tree-0.1.5-py3-none-any.whl", "ascii-tree"),
        ("Jinja2-3.1.6-py3-none-any.whl", "jinja2"),
        ("zope.interface-8.0-py3-none-any.whl", "zope-interface"),
    ],
)
def test_canonical_distribution_normalizes_per_pep503(filename, expected):
    assert pkg.canonical_distribution(Path(filename)) == expected


def test_install_environment_wheels_survives_a_duplicated_distribution(
    tmp_path, monkeypatch
):
    """End to end: the exact wheelhouse shape that failed on the cluster now installs."""
    env_root = _make_package(
        tmp_path / "environment",
        wheels=(
            "ascii_tree-0.1.5-py3-none-any.whl",
            "xxhash-3.8.1-py3-none-any.whl",
            "xxhash-4.0.0-py3-none-any.whl",
        ),
    )
    venv_root = tmp_path / "gym_venvs"
    _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")

    calls = []
    monkeypatch.setattr(
        pkg.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _ok()
    )

    pkg.install_environment_wheels({pkg.UV_VENV_DIR_KEY: str(venv_root)}, str(env_root))

    (cmd,) = calls
    assert "ascii-tree" in cmd
    assert "xxhash" in cmd
    assert not any(c.startswith("xxhash==") for c in cmd)


def test_install_environment_wheels_surfaces_the_resolver_error(tmp_path, monkeypatch):
    """A failed install must report WHY, not just that a subprocess exited non-zero.

    CalledProcessError renders the whole ~100-element argv, pushing the one useful line
    (`Because openai-agents==0.20.0 depends on openai>=2.45.0,<3 ...`) off the top of the
    traceback -- and that line is the entire diagnosis.
    """
    env_root = _make_package(tmp_path / "environment")
    venv_root = tmp_path / "gym_venvs"
    _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")

    monkeypatch.setattr(
        pkg.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=1,
            stderr="  x No solution found when resolving dependencies:\n"
            "  |-> Because openai-agents==0.20.0 depends on openai>=2.45.0,<3",
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        pkg.install_environment_wheels(
            {pkg.UV_VENV_DIR_KEY: str(venv_root)}, str(env_root)
        )

    message = str(excinfo.value)
    assert "openai-agents==0.20.0 depends on openai" in message
    assert "rebuild it rather than relaxing the install" in message


def test_replaced_distributions_reads_the_uv_summary():
    """uv reports a swap as `- name==old` / `+ name==new`; only removals are dangerous."""
    summary = (
        "Resolved 99 packages in 24ms\n"
        " - protobuf==6.33.6\n"
        " + protobuf==7.35.1\n"
        " + ascii-tree==0.1.5\n"
    )
    assert pkg.replaced_distributions(summary) == ["protobuf"]


def test_install_warns_when_it_replaces_a_live_package(tmp_path, monkeypatch, capsys):
    """The servers are already running, so a replacement fails later, not here.

    Real case: forcing protobuf 6.33.6 -> 7.35.1 under a live uvicorn produced
    `VersionError: gencode 7.35.1 runtime 6.33.6` on the first POST /run, with nothing
    tying it back to this install.
    """
    env_root = _make_package(tmp_path / "environment")
    venv_root = tmp_path / "gym_venvs"
    _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")

    monkeypatch.setattr(
        pkg.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stderr=" - protobuf==6.33.6\n + protobuf==7.35.1\n"
        ),
    )

    pkg.install_environment_wheels({pkg.UV_VENV_DIR_KEY: str(venv_root)}, str(env_root))

    out = capsys.readouterr().out
    assert "replaced 1 package(s)" in out
    assert "protobuf" in out
    assert "may fail at the first rollout" in out


def test_install_is_quiet_when_nothing_is_replaced(tmp_path, monkeypatch, capsys):
    """The expected path: unpinned requirements only add what is missing."""
    env_root = _make_package(tmp_path / "environment")
    venv_root = tmp_path / "gym_venvs"
    _make_server_venv(venv_root, "responses_api_agents", "verifiers_agent")

    monkeypatch.setattr(
        pkg.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stderr=" + ascii-tree==0.1.5\n"
        ),
    )

    pkg.install_environment_wheels({pkg.UV_VENV_DIR_KEY: str(venv_root)}, str(env_root))

    assert "WARNING replaced" not in capsys.readouterr().out
