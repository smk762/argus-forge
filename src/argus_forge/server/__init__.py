"""argus-forge HTTP server (optional ``server`` extra)."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from argus_forge.server.app import create_app, env_readonly

__all__ = ["create_app", "env_readonly"]


def __getattr__(name: str) -> Any:
    """Resolve the FastAPI entry points lazily.

    Importing :mod:`argus_forge.server.app` here eagerly would mean that
    ``import argus_forge.server.jobs`` — which needs nothing from the ``server``
    extra — first pulled in fastapi, starlette and argus-cortex, since importing
    a submodule runs its package's ``__init__`` first. That is the coupling
    putting the registry under ``server/`` was meant to avoid, and it made the
    registry's HTTP-free unit tests unrunnable on an install without that extra.
    """
    if name in __all__:
        from argus_forge.server import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
