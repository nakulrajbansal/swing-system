"""Entry point for the desktop app.

    python -m app.main                 # launch the GUI
    python -m app.main --selftest      # headless: run a fast validation, no display
    python -m app.main --selftest paper

The self-test path lets the packaged executable be verified in CI / on a build
box that has no display, by running the real backend headlessly.
"""

from __future__ import annotations

import os
import sys
import tempfile

from app.config import AppConfig
from app.runner import run_paper, run_validation


def _selftest(which: str) -> int:
    # A small, fast universe so the bundle can be verified quickly.
    cfg = AppConfig(n_symbols=4, start_date="2021-01-04", end_date="2021-12-31",
                    seed=1, oos_start="2021-09-01")
    fn = run_paper if which == "paper" else run_validation

    # Always log to a file (works in windowed builds where sys.__stdout__ is None),
    # and also to the real stdout when a console exists. The runner redirects
    # sys.stdout to `emit`, so we must write to the REAL streams to avoid recursion.
    log_path = os.path.join(tempfile.gettempdir(), "swing_selftest.log")
    logf = open(log_path, "w", encoding="utf-8")
    real = sys.__stdout__

    def emit(line: str) -> None:
        logf.write(line + "\n")
        logf.flush()
        if real is not None:
            real.write(line + "\n")
            real.flush()

    try:
        emit(f"[selftest] running {fn.__name__} on a small synthetic universe...")
        fn(cfg, emit)
        emit("[selftest] OK")
        return 0
    except Exception as exc:
        emit(f"[selftest] FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        emit(f"[selftest] log written to {log_path}")
        logf.close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        i = argv.index("--selftest")
        which = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else "validation"
        return _selftest(which)
    from app.gui import launch          # imported lazily so --selftest needs no display
    launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
