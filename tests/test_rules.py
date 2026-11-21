"""Checks behind CLAUDE.md's "Must never happen" table that no single feature test owns.

Each test guards a rule about the *shape* of the code, so that a change which
breaks the rule fails here rather than relying on someone remembering it in review.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import argus_cortex
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from argus_forge.server import create_app
from argus_forge.server.app import TRAINING_DISABLED_DETAIL

SRC = Path(__file__).resolve().parent.parent / "src" / "argus_forge"
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}

# Write-method routes that stay open in read-only mode, each with the reason it
# writes nothing there. Anything else that takes an unsafe method must live
# under /run, where the read-only middleware refuses it by path.
READONLY_OPEN_WRITES: dict[str, str] = {
    "/inspect": "reads the export, writes nothing",
    "/config": "forced to dry_run: renders and returns files, writes nothing",
}


def _write_routes(app) -> list[APIRoute]:
    return [r for r in app.routes if isinstance(r, APIRoute) and r.methods & UNSAFE]


def test_every_write_route_is_refused_or_known_safe_under_readonly(tmp_path: Path) -> None:
    """Walk the app: under read-only, every unsafe route is a refused /run route or a listed safe one.

    A new write route outside /run (a /train, an /upload) fails here until it is
    moved under /run or added to READONLY_OPEN_WRITES with the reason it is safe.
    """
    app = create_app(export_root=str(tmp_path), allow_run=False)
    routes = _write_routes(app)
    assert routes, "no write routes found: the walk is broken"
    with TestClient(app) as client:
        for route in routes:
            if route.path == "/run" or route.path.startswith("/run/"):
                path = re.sub(r"\{[^}]+\}", "x", route.path)
                for method in route.methods & UNSAFE:
                    # A body that could never validate: the refusal must land before validation.
                    resp = client.request(method, path, content=b"not json")
                    assert resp.status_code == 403, f"{method} {route.path} is not refused under read-only"
                    assert resp.json()["detail"] == TRAINING_DISABLED_DETAIL
            else:
                assert route.path in READONLY_OPEN_WRITES, (
                    f"{sorted(route.methods & UNSAFE)} {route.path} is a write route outside /run: move it under "
                    "/run so read-only refuses it, or add it to READONLY_OPEN_WRITES with why it writes nothing"
                )
    open_paths = {r.path for r in routes}
    assert set(READONLY_OPEN_WRITES) <= open_paths, "READONLY_OPEN_WRITES names a route that no longer exists"


def _cortex_public_callables() -> dict[str, str]:
    """Every public function and class argus-cortex defines, by name, with the module that defines it."""
    found: dict[str, str] = {}
    for info in pkgutil.iter_modules(argus_cortex.__path__):
        if info.name.startswith("_") or info.name == "codemods":
            continue
        try:
            module = importlib.import_module(f"argus_cortex.{info.name}")
        except ImportError:  # an optional extra this env lacks
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or not callable(obj):
                continue
            if getattr(obj, "__module__", "") == module.__name__:
                found.setdefault(name, module.__name__)
    return found


def _delegates(node: ast.AST, cortex_module: str) -> bool:
    """Whether *node*'s body imports the same-named function from *cortex_module* (a thin wrapper, not a copy)."""
    name = node.name  # type: ignore[attr-defined]
    return any(
        isinstance(n, ast.ImportFrom) and n.module == cortex_module and any(a.name == name for a in n.names)
        for n in ast.walk(node)
    )


def test_forge_does_not_reimplement_cortex() -> None:
    """No function or class in forge shares a name with one argus-cortex already provides.

    The suite's shared code (taxonomy, wire-schema building, the write guard and
    env flags, the GPU lease, path and URL safety) lives in cortex. A same-named
    definition here is almost always a copy that will drift. Import it instead.
    A wrapper that keeps the name for forge's callers is allowed only when its own
    body imports the cortex original (``models.wire_schema`` does).
    """
    cortex = _cortex_public_callables()
    assert "WriteGuard" in cortex and "wire_schema" in cortex, "cortex scan is broken"
    clashes = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "_version.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.name in cortex:
                if _delegates(node, cortex[node.name]):
                    continue
                rel = path.relative_to(SRC.parent)
                clashes.append(f"{rel}:{node.lineno} {node.name} (cortex has {cortex[node.name]}.{node.name})")
    assert not clashes, "forge re-implements argus-cortex:\n" + "\n".join(clashes)


# Imports that would mean forge measures or compares poses itself, or calls a service
# (argus-puppet) rather than reading the manifest curator wrote. Forge's own HTTP need,
# the GPU broker, goes through argus-cortex and is not imported here.
POSE_FORBIDDEN_IMPORTS = {"argus_puppet", "numpy", "scipy", "sklearn", "httpx", "requests", "urllib", "aiohttp"}


def test_forge_does_not_compute_pose_metrics_or_call_puppet() -> None:
    """The pose-dominance warning only counts curator's ``pose_cluster`` / ``pose_labels``.

    Pose embeddings, distances and clusters are argus-puppet's; a copy here would drift
    from it. So no forge module imports puppet, a numeric stack, or an HTTP client, and
    nothing names a puppet endpoint or setting.
    """
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "_version.py":
            continue
        rel = path.relative_to(SRC.parent)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".")[0]}
            else:
                roots = set()
            found += [f"{rel}:{node.lineno} imports {r}" for r in sorted(roots & POSE_FORBIDDEN_IMPORTS)]
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.search(r"PUPPET_URL|ARGUS_PUPPET|/pose\b", node.value)
            ):
                found.append(f"{rel}:{node.lineno} names a puppet endpoint or setting")
    assert not found, "forge must read pose from curator's manifest, not compute it:\n" + "\n".join(found)
