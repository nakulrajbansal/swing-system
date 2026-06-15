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
    if which == "live":
        # Tiny REAL-data validation: proves yfinance works inside the bundle.
        cfg = AppConfig(data_source="live", n_symbols=2,
                        start_date="2022-01-01", end_date="2022-12-31",
                        oos_start="2022-09-01")
        fn = run_validation
    else:
        # A small, fast synthetic universe so the bundle can be verified quickly.
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


def _headless(fn, log_name: str, **overrides) -> int:
    """Run one desk action with the SAVED config, no display — the scheduled
    daily screen / exit review path. Output goes to a temp log and (when a
    console exists) stdout; the runner also tees to ~/.swing_system/logs."""
    import dataclasses

    cfg = AppConfig.load()
    for k, v in overrides.items():
        cfg = dataclasses.replace(cfg, **{k: v})
    log_path = os.path.join(tempfile.gettempdir(), f"swing_{log_name}.log")
    logf = open(log_path, "w", encoding="utf-8")
    real = sys.__stdout__

    def emit(line: str) -> None:
        logf.write(line + "\n")
        logf.flush()
        if real is not None:
            real.write(line + "\n")
            real.flush()

    try:
        emit(f"[headless] {fn.__name__} starting (config: ~/.swing_system/config.json)")
        fn(cfg, emit)
        emit(f"[headless] {fn.__name__} OK")
        return 0
    except Exception as exc:
        emit(f"[headless] FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        logf.close()


_SCREEN_KEYS = {"sp500", "qqq", "sp400", "sp600", "midsmall", "broad"}


def _arg_after(argv, flag):
    i = argv.index(flag)
    return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else None


def _screen_universes(argv) -> list[str]:
    """The universe(s) a headless screen runs: an explicit `--screen a,b` arg,
    else the configured `scheduled_screen_universes`. Flexible & comma-list."""
    from app.schedule import screen_universes
    arg = _arg_after(argv, "--screen") if "--screen" in argv else None
    if arg:
        picks = [u.strip() for u in arg.split(",")]
        return [u for u in picks if u in _SCREEN_KEYS] or screen_universes(AppConfig.load())
    return screen_universes(AppConfig.load())


def _headless_screens(argv) -> int:
    """Screen one or more universes headlessly (cloud/scheduled)."""
    from app.runner import run_screen
    rc = 0
    for uni in _screen_universes(argv):
        rc = _headless(run_screen, f"screen-{uni}", screen_index=uni) or rc
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        i = argv.index("--selftest")
        which = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else "validation"
        return _selftest(which)
    if "--screen" in argv:
        return _headless_screens(argv)
    if "--review" in argv:
        from app.runner import run_position_review
        return _headless(run_position_review, "review")
    if "--daily" in argv:
        # The scheduled combined run: manage exits first, then screen each
        # configured universe. (Manage-exits-only: the screen never auto-buys
        # unless place_orders is explicitly enabled.)
        from app.runner import run_position_review
        rc = _headless(run_position_review, "review")
        return _headless_screens(argv) or rc
    if "--watch" in argv:
        from app.runner import run_watch
        return _headless(run_watch, "watch")
    from app.gui import launch          # imported lazily so --selftest needs no display
    launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
