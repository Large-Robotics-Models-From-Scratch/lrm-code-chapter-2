"""ch02 — Chapter 2 export contract for downstream chapters.

Chapter 3 imports `make_pickplace_dataloader`, `normalize`, `denormalize`,
and `StatsDict` directly from this package root. These names are frozen;
renaming or re-ordering arguments breaks every downstream chapter.
`make_env` is re-exported for eval-rollout chapters.

All re-exports are lazy (PEP 562 `__getattr__`) so `import ch02` doesn't
trigger the [sim] or [data] extras until a symbol is actually accessed.
"""

__all__ = [
    "StatsDict",
    "denormalize",
    "make_env",
    "make_pickplace_dataloader",
    "normalize",
]

_PIPELINE_EXPORTS = {
    "StatsDict",
    "denormalize",
    "make_pickplace_dataloader",
    "normalize",
}


def __getattr__(name):
    if name == "make_env":
        from ch02.env import make_env
        return make_env
    if name in _PIPELINE_EXPORTS:
        from ch02 import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'ch02' has no attribute {name!r}")
