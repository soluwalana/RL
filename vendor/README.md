# Vendored wheels

## `nemo_sandboxed_gym-0.1.0rc0-py3-none-any.whl`

A local build of `packages/sandboxed_gym` from
[nemo-platform](https://github.com/NVIDIA-NeMo/nemo-platform), vendored until that package is
published to PyPI.

It is here, rather than referenced as a sibling checkout, because `[tool.uv.sources]` paths
resolve relative to `pyproject.toml`. In the training image that is `/opt/nemo-rl`, and only
this repository enters the build context — so a `../nemo-platform/...` path resolves to a
directory that does not exist, and the `SandboxedGymActor` venv fails to build.

**Temporary.** `pyproject.toml` already requires `nemo-sandboxed-gym==0.1.0rc0`, the version
this wheel carries. Once that release is on PyPI, delete the `[tool.uv.sources]` entry and this
directory; nothing else has to change.

Rebuild from a nemo-platform checkout with:

```bash
UV_DYNAMIC_VERSIONING_BYPASS=0.1.0rc0 uv build --package nemo-sandboxed-gym -o <dest>
```
