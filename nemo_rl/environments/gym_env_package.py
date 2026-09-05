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

"""Make a staged environment package usable by the Gym servers that run it.

An environment package reaches NeMo-RL as a directory path (``env.nemo_gym.environment_path``).
Gym then needs two things it does not do on its own: the package on its component search
path, so ``native-v1`` server trees resolve by name, and the vendored ``wheels/`` closure
installed into the per-server venvs, which Gym builds from framework-only requirements.

Shared by the colocated actor (:mod:`nemo_rl.environments.nemo_gym`) and the in-sandbox
host (:mod:`nemo_rl.environments.sandbox.gym_host_runtime`). Imports only the standard
library: neither caller is guaranteed the sandbox package's dependencies.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

# Gym's own component search-path variable (nemo_gym.component_search_roots).
NEMO_GYM_EXTRA_ROOTS_ENV_VAR = "NEMO_GYM_EXTRA_ROOTS"
# uv reads this natively, which is how the wheelhouse reaches Gym's own install step.
UV_FIND_LINKS_ENV_VAR = "UV_FIND_LINKS"
# uv delimits --find-links with ",", unlike NEMO_GYM_EXTRA_ROOTS above, which Gym splits
# on os.pathsep.
UV_FIND_LINKS_SEPARATOR = ","
# Makes uv ignore pyproject.toml / uv.toml discovery; honoured by every subcommand.
UV_NO_CONFIG_ENV_VAR = "UV_NO_CONFIG"
# --no-index has no env var; UV_OFFLINE is the only one uv exposes here.
UV_OFFLINE_ENV_VAR = "UV_OFFLINE"
# Gym's global-config key for the root the per-server venvs are built under.
UV_VENV_DIR_KEY = "uv_venv_dir"
# Server types that run user environment code. Model servers are excluded: they proxy to
# vLLM and never import the environment package. Resources servers are included because a
# wheels-v1 package declares no adapter, so its code may be a verifier rather than an agent.
ENV_PACKAGE_SERVER_TYPES = ("responses_api_agents", "resources_servers")


def register_environment_search_root(env_root: str | None) -> str | None:
    """Prepend an environment package to Gym's component search path.

    Prepended so a user's environment shadows a built-in of the same name. Must run before
    ``RunHelper.start`` resolves server directories, and before ``nemo_gym`` is imported if
    the caller needs the root on ``sys.path`` (``_augment_sys_path`` runs at import time).

    Args:
        env_root: Staging directory of the environment package, or None/"" when there is none.

    Returns:
        The resulting search path, or None when there was nothing to register.

    Raises:
        FileNotFoundError: ``env_root`` is set but is not a directory. Gym would otherwise
            fall back to a built-in server of the same name and train the wrong environment.
    """
    root = (env_root or "").strip()
    if not root:
        return None
    if not Path(root).is_dir():
        raise FileNotFoundError(
            f"Environment package root {root!r} is not a directory. It is registered as a "
            "Gym component search root, and Gym silently falls back to a built-in server of "
            "the same name when a root does not resolve -- which would train against the "
            "wrong environment without erroring. Check that the platform staged the "
            "environment FileSet to this path."
        )
    existing = os.environ.get(NEMO_GYM_EXTRA_ROOTS_ENV_VAR, "")
    entries = [root, *(e for e in existing.split(os.pathsep) if e and e != root)]
    joined = os.pathsep.join(entries)
    os.environ[NEMO_GYM_EXTRA_ROOTS_ENV_VAR] = joined
    print(f"NeMo Gym: environment search root -> {root}")
    return joined


def isolate_uv_from_ambient_project() -> None:
    """Stop uv applying the ``[tool.uv]`` dependency pins of the project owning the CWD.

    uv discovers those from the working directory and applies them to every invocation,
    so an environment package's venvs would otherwise resolve against the host project's
    constraints. Set process-wide because Gym composes its own uv commands.
    """
    os.environ.setdefault(UV_NO_CONFIG_ENV_VAR, "1")


def environment_wheels_dir(env_root: str | None) -> Path | None:
    """Return the package's ``wheels/`` directory when it holds at least one wheel."""
    root = (env_root or "").strip()
    if not root:
        return None
    wheels_dir = Path(root) / "wheels"
    return wheels_dir if wheels_dir.is_dir() and any(wheels_dir.glob("*.whl")) else None


def configure_environment_wheelhouse(env_root: str | None, *, offline: bool = False) -> Path | None:
    """Add an environment package's vendored wheels to uv's resolution sources.

    Must run before ``RunHelper.start``: Gym composes the per-server ``uv pip install``
    itself, so uv's environment is the only way to reach it.

    ``offline`` also sets ``UV_OFFLINE``, for a wheelhouse the caller knows to be
    self-sufficient. Without it, an index that is configured but unreachable makes uv fail to
    resolve the ``uv venv --seed`` packages rather than fall back to ``--find-links``. The
    caller decides, because a package can ship wheels and still need an index for its agent.

    Args:
        env_root: Staging directory of the environment package, or None/"" when there is none.
        offline: Resolve from the wheelhouse and the uv cache only.

    Returns:
        The wheelhouse directory, or None when the package ships no wheels.
    """
    wheels_dir = environment_wheels_dir(env_root)
    if wheels_dir is None:
        return None

    existing = os.environ.get(UV_FIND_LINKS_ENV_VAR, "")
    entries = [
        str(wheels_dir),
        *(
            e
            for e in existing.split(UV_FIND_LINKS_SEPARATOR)
            if e and e != str(wheels_dir)
        ),
    ]
    os.environ[UV_FIND_LINKS_ENV_VAR] = UV_FIND_LINKS_SEPARATOR.join(entries)
    print(
        f"NeMo Gym: environment wheelhouse -> {wheels_dir} "
        f"({len(list(wheels_dir.glob('*.whl')))} wheel(s))"
    )

    if offline:
        # An explicit operator setting wins.
        if UV_OFFLINE_ENV_VAR in os.environ:
            print(
                f"NeMo Gym: offline requested, but {UV_OFFLINE_ENV_VAR}="
                f"{os.environ[UV_OFFLINE_ENV_VAR]} is already set"
            )
        else:
            os.environ[UV_OFFLINE_ENV_VAR] = "1"
            print(f"NeMo Gym: offline wheelhouse, setting {UV_OFFLINE_ENV_VAR}=1")
    return wheels_dir


def _wheel_name_version(wheel: Path) -> tuple[str, str]:
    """Split a PEP 427 wheel filename into its distribution and version fields."""
    # {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl, and a distribution
    # never contains "-" (it is escaped to "_"), so the first two fields are unambiguous.
    parts = wheel.name.split("-")
    if len(parts) < 5:
        raise ValueError(f"Not a PEP 427 wheel filename: {wheel.name}")
    return parts[0], parts[1]


def canonical_distribution(wheel: Path) -> str:
    """Return a wheel's PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", _wheel_name_version(wheel)[0]).lower()


def wheel_requirement(wheel: Path) -> str:
    """Return ``name==version`` for a wheel, parsed from its PEP 427 filename."""
    name, version = _wheel_name_version(wheel)
    return f"{name}=={version}"


def wheelhouse_requirements(wheels: list[Path]) -> list[str]:
    """Turn a wheelhouse into requirements: one bare distribution name per project.

    Unpinned on purpose. A wheelhouse is a candidate pool, that says "here is everything
    you might need, offline" not a lock file for the venv which forces  that "the venv must
    look exactly like this": it was resolved against the environment alone, while Gym resolved
    the venv from the server's own requirements. Pinning its versions would overwrite packages the
    running server has already imported, which fails at the first rollout rather than here.
    Unpinned, an already-satisfied requirement is a no-op and only missing packages install.
    """
    return sorted({canonical_distribution(wheel) for wheel in wheels})


def env_package_venv_pythons(venv_root: str) -> list[Path]:
    """Return the interpreters of the venvs Gym built for env-code-running servers.

    Gym lays these out as ``<uv_venv_dir>/<server_type>/<name>/.venv`` (``setup_env_command``
    in nemo_gym.cli.setup_command), discoverable by glob once ``RunHelper.start`` returns.
    Globbed rather than derived from the merged config so a venv cannot be missed; an image
    with prefetched venvs may therefore receive the closure for servers it never starts.
    """
    return [
        python
        for server_type in ENV_PACKAGE_SERVER_TYPES
        for venv in sorted(Path(venv_root).glob(f"{server_type}/*/.venv"))
        if (python := venv / "bin" / "python").is_file()
    ]


def replaced_distributions(install_summary: str) -> list[str]:
    """Distributions the installer removed to make room for another version.

    uv reports a change as a ``- name==old`` / ``+ name==new`` pair. Only removals matter:
    an addition is inert, a removal rewrites a package the server may already have imported.
    """
    return sorted(
        {
            line.strip()[2:].split("==")[0]
            for line in install_summary.splitlines()
            if line.strip().startswith("- ") and "==" in line
        }
    )


def _warn_on_replaced_packages(install_summary: str, python: Path) -> None:
    """Flag packages swapped underneath an already-running server process.

    Gym starts the servers during ``RunHelper.start``, before this runs, so a replacement
    changes files under a live interpreter and surfaces only at the first rollout.
    Requirements are unpinned so this normally finds nothing.
    """
    replaced = replaced_distributions(install_summary)
    if not replaced:
        return
    print(
        f"NeMo Gym: WARNING replaced {len(replaced)} package(s) already installed in "
        f"{python}: {', '.join(replaced)}. The Gym server running from this venv started "
        "before the install, so anything it had already imported now disagrees with the "
        "files on disk and may fail at the first rollout. This means the environment "
        "package needs a version the agent framework had already resolved differently; "
        "regenerate the package resolved against the agent's own requirements."
    )


def install_environment_wheels(
    global_config: Mapping[str, Any], env_root: str | None
) -> None:
    """Install an environment package's vendored closure into the server venvs.

    Installed offline from the staged ``wheels/`` directory, so spin-up works on a sandbox
    whose egress cannot reach the index the package was published to. Must run after
    ``RunHelper.start``, which is when the per-server venvs are created.

    Every failure raises: a missing environment package would otherwise surface at the first
    rollout as an opaque import error.

    Args:
        global_config: Gym's global config dict; read for ``uv_venv_dir``.
        env_root: Staging directory of the environment package, or None/"" when there is none.

    Raises:
        RuntimeError: no ``uv_venv_dir`` configured, no server venv found, or the install
            failed.
    """
    root = (env_root or "").strip()
    if not root:
        return
    wheels_dir = Path(root) / "wheels"
    wheels = sorted(wheels_dir.glob("*.whl")) if wheels_dir.is_dir() else []
    if not wheels:
        # native-v1 packages and bundled config_paths runs carry no wheels; both are valid.
        print(f"NeMo Gym: no wheels under {wheels_dir}, nothing to install")
        return

    venv_root = global_config.get(UV_VENV_DIR_KEY)
    if not venv_root:
        # Gym puts venvs at <server_dir>/.venv when NEMO_GYM_VENV_DIR is unset, and that
        # path is not derivable here. Skipping the install would fail at the first rollout.
        raise RuntimeError(
            f"{len(wheels)} environment wheel(s) under {wheels_dir}, but Gym has no "
            f"{UV_VENV_DIR_KEY!r} configured, so the per-server venvs cannot be located. "
            "Set NEMO_GYM_VENV_DIR (the container images do) so the environment package "
            "can be installed into them."
        )

    pythons = env_package_venv_pythons(str(venv_root))
    if not pythons:
        raise RuntimeError(
            f"{len(wheels)} environment wheel(s) under {wheels_dir}, but Gym built no "
            f"server venv under {venv_root!r} for any of {ENV_PACKAGE_SERVER_TYPES}. "
            "Installing nothing would leave the environment package missing until the "
            "first rollout fails to import it."
        )

    requirements = wheelhouse_requirements(wheels)
    for python in pythons:
        print(
            f"NeMo Gym: installing {len(requirements)} environment package(s) into {python}"
        )
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            # Explicit as well as the env var: this resolve must see only the wheelhouse.
            "--no-config",
            "--no-index",
            f"--find-links={wheels_dir}",
            *requirements,
        ]
        # Captured so a failure can be re-raised with the resolver's explanation rather
        # than a CalledProcessError repr of the whole argv.
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        summary = (result.stderr or "").strip()
        if result.returncode != 0:
            raise RuntimeError(
                f"Installing the environment package into {python} failed.\n{summary}\n"
                "The vendored closure under wheels/ is what the resolver is working from, "
                "so a conflict here means the package ships a set that cannot be installed "
                "together -- rebuild it rather than relaxing the install."
            )
        if summary:
            # uv reports what it changed on stderr; the only record of the venv contents.
            print(summary)
        _warn_on_replaced_packages(summary, python)
