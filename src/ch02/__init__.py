"""Chapter 2 public exports. Pipeline symbols join here in §2.5."""

__all__ = ["make_env"]


def __getattr__(name):
    if name == "make_env":
        from ch02.env import make_env
        return make_env
    raise AttributeError(f"module 'ch02' has no attribute {name!r}")
