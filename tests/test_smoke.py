"""Smoke tests for the ch02 package.

Verifies the package installs and imports cleanly. Real per-module
coverage lands in tests/test_pipeline.py (PR 6), tests/test_scripted.py
(PR 3), etc. — this file exists so CI has something to collect before
any chapter listings are written.
"""


def test_ch02_imports():
    import ch02

    assert ch02 is not None
