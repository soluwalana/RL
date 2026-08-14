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

"""Make a platform environment package usable by the Gym servers that run it.

A NeMo Platform GRPO job stages one environment FileSet to a directory and hands
NeMo-RL the path (``env.nemo_gym.environment_path``); NeMo-RL never sees FileSets. Two
things then have to happen before a rollout can succeed, and Gym does neither on its own:

1. **Search root.** A ``native-v1`` package ships its own server tree
   (``resources_servers/<name>/``). Gym resolves a server directory by NAME against
   ``nemo_gym.component_search_roots()``, and a staged package is on none of those roots.
2. **Wheel closure.** ``wheels-v1`` / ``adapter-wheels-v1`` packages vendor their
   dependency closure under ``wheels/``. Gym builds each server venv from that server's
   own requirements, which carry the framework only, so the environment package itself is
   never installed.

Both the colocated actor (:mod:`nemo_rl.environments.nemo_gym`) and the in-sandbox host
(:mod:`nemo_rl.environments.sandbox.gym_host_runtime`) need identical behaviour here, so
it lives in one place rather than being mirrored in two.

Deliberately imports only the standard library. The colocated actor's venv is not
guaranteed to have fastapi or ``nemo_gym.sandbox.broker`` (which
``nemo_rl.environments.sandbox.__init__`` pulls in eagerly), and the sandbox host runs as
a script from the image tree, so anything either side imports has to be this cheap.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

# Gym's own component search-path variable (nemo_gym.component_search_roots).
NEMO_GYM_EXTRA_ROOTS_ENV_VAR = "NEMO_GYM_EXTRA_ROOTS"
# Gym's global-config key for the root the per-server venvs are built under.
UV_VENV_DIR_KEY = "uv_venv_dir"
# Server types that run user environment code. Model servers are excluded: they proxy to
# vLLM and never import the environment package. Resources servers are included because a
# wheels-v1 package declares no adapter, so its code may be a verifier rather than an agent.
ENV_PACKAGE_SERVER_TYPES = ("responses_api_agents", "resources_servers")


def register_environment_search_root(env_root: str | None) -> str | None:
    """Put an environment package on Gym's component search path.

    Gym resolves a server directory by NAME against ``component_search_roots()`` -- extra
    roots, then sys.path, cwd, and the Gym install root. A staged package is on none of
    those, so without this a ``native-v1`` package is staged and never used.

    Prepended rather than appended, because earlier roots win on a name collision -- which
    is what lets a user's environment shadow a built-in of the same name.

    Must run BEFORE ``RunHelper.start``, which is when server directories are resolved,
    and before ``nemo_gym`` is first imported if the importing process itself needs the
    root on ``sys.path`` (``nemo_gym._augment_sys_path`` runs at import time). Gym re-reads
    the variable at call time and ``run_command`` copies the environment into every server
    subprocess, so setting it once here reaches them all.

    Args:
        env_root: Staging directory of the environment package, or None/"" when there is
            none (standalone NeMo-RL, or a run using only bundled ``config_paths``).

    Returns:
        The resulting search path, or None when there was nothing to register.

    Raises:
        FileNotFoundError: ``env_root`` is set but is not a directory. Registering a
            non-existent root is worse than failing: Gym would fall through to its install
            root and silently start a BUILT-IN server of the same name, so the job would
            train to completion against the wrong environment and report success.
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


def _wheel_name_version(wheel: Path) -> tuple[str, str]:
    """Split a PEP 427 wheel filename into its distribution and version fields."""
    # {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl, and a distribution
    # never contains "-" (it is escaped to "_"), so the first two fields are unambiguous.
    parts = wheel.name.split("-")
    if len(parts) < 5:
        raise ValueError(f"Not a PEP 427 wheel filename: {wheel.name}")
    return parts[0], parts[1]


def canonical_distribution(wheel: Path) -> str:
    """Return a wheel's PEP 503 normalized distribution name.

    Wheel filenames escape the distribution with ``_``; the installer wants the normalized
    form, so ``charset_normalizer-3.5.0-...`` and a requirement for ``charset-normalizer``
    are recognised as the same project.
    """
    return re.sub(r"[-_.]+", "-", _wheel_name_version(wheel)[0]).lower()


def wheel_requirement(wheel: Path) -> str:
    """Return ``name==version`` for a wheel, parsed from its PEP 427 filename.

    Requirements rather than wheel paths: ``uv pip install <path>.whl`` uninstalls and
    reinstalls the package even when that exact version is already present, so passing
    paths would rebuild the whole dependency tree of every server venv on each spin-up.
    Names resolve against ``--find-links`` and leave already-satisfied packages alone.
    """
    name, version = _wheel_name_version(wheel)
    return f"{name}=={version}"


def wheelhouse_requirements(wheels: list[Path]) -> list[str]:
    """Turn a wheelhouse into requirements to install, one per distribution.

    Pinning every FILE is wrong when a wheelhouse carries two versions of the same
    distribution: the installer is then handed ``xxhash==3.8.1`` and ``xxhash==4.0.0`` in
    one command and fails as unsatisfiable, no matter which one the environment needs.
    Several versions of a project in a ``--find-links`` directory is normal -- it is a
    candidate pool, not a lock file -- and a producer that resolves while downloading can
    leave the versions it backtracked away from behind.

    So: pin a distribution only where the wheelhouse is unambiguous (exact and
    reproducible, and a no-op when that version is already installed), and fall back to
    the bare name where it is not, which is the most the pool can actually say. The
    ambiguity is logged rather than swallowed, because it points at a producer bug.
    """
    by_distribution: dict[str, list[Path]] = {}
    for wheel in wheels:
        by_distribution.setdefault(canonical_distribution(wheel), []).append(wheel)

    requirements: list[str] = []
    for distribution, group in sorted(by_distribution.items()):
        if len(group) == 1:
            requirements.append(wheel_requirement(group[0]))
            continue
        versions = sorted(_wheel_name_version(wheel)[1] for wheel in group)
        print(
            f"NeMo Gym: WARNING the environment package ships {len(group)} versions of "
            f"{distribution} ({', '.join(versions)}); requesting it unpinned so the "
            "resolver picks one. Regenerate the package so it vendors a single resolved "
            "closure if you need the installed version to be reproducible."
        )
        requirements.append(distribution)
    return requirements


def env_package_venv_pythons(venv_root: str) -> list[Path]:
    """Return the interpreters of the venvs Gym built for env-code-running servers.

    Gym lays these out as ``<uv_venv_dir>/<server_type>/<name>/.venv`` (see
    ``setup_env_command`` in nemo_gym.cli.setup_command), so they are discoverable by glob
    once ``RunHelper.start`` has returned.

    Globbing rather than deriving the list from the merged Gym config on purpose: a glob
    cannot MISS a venv, and a missed venv surfaces as an opaque import error at the first
    rollout. The cost is that an image built with ``NEMO_GYM_PREFETCH_CONFIGS`` set (the
    Nemotron recipes; not the platform default, which bakes none) also carries venvs this
    job never starts, and they receive the closure too -- wasted work, and a hard failure
    if one of them pins a conflicting version. Prefer over-installing to under-installing:
    the former fails loudly here, the latter fails silently much later.
    """
    return [
        python
        for server_type in ENV_PACKAGE_SERVER_TYPES
        for venv in sorted(Path(venv_root).glob(f"{server_type}/*/.venv"))
        if (python := venv / "bin" / "python").is_file()
    ]


def install_environment_wheels(
    global_config: Mapping[str, Any], env_root: str | None
) -> None:
    """Install an environment package's vendored closure into the server venvs.

    A ``wheels-v1`` / ``adapter-wheels-v1`` package ships its own dependency closure under
    ``wheels/``. Gym builds each server venv from that server's own requirements, which
    carry the framework only, so the environment package is installed here instead --
    offline, from the staged directory, with no package index involved. That keeps spin-up
    working on a sandbox whose egress does not reach the index the package was published to.

    Must run AFTER ``RunHelper.start``: the per-server venvs do not exist until it returns.
    Safe there because an agent resolves its environment lazily on the first rollout
    (``_get_env`` in ``verifiers_agent/app.py``), so the packages only have to be importable
    before the first rollout, not at spin-up.

    Every failure raises. A missing environment package surfaces at the first rollout as an
    opaque ``load_environment`` error, long after the cause, so anything that would leave
    the closure uninstalled must fail here instead.

    Args:
        global_config: Gym's global config dict; read for ``uv_venv_dir``.
        env_root: Staging directory of the environment package, or None/"" when there is none.
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
        # path is not derivable here. Raise rather than warn-and-continue: we only get here
        # when there ARE wheels, so skipping them guarantees the run dies at the first
        # rollout instead -- with an ImportError that says nothing about this.
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
            "--no-index",
            f"--find-links={wheels_dir}",
            *requirements,
        ]
        # stderr is captured so a failure can be re-raised with the resolver's own
        # explanation. Left to CalledProcessError, the traceback ends in a repr of the
        # whole ~100-element argv and the one line that says WHY scrolls off above it.
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Installing the environment package into {python} failed.\n"
                f"{(result.stderr or '').strip()}\n"
                "The vendored closure under wheels/ is what the resolver is working from, "
                "so a conflict here means the package ships a set that cannot be installed "
                "together -- rebuild it rather than relaxing the install."
            )
