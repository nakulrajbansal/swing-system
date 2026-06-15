"""Scheduling presets — the single source of truth for WHEN the desk runs.

Times are reasoned in US Eastern (the exchange's timezone) and converted to the
local machine (for Windows Task Scheduler) or to UTC (for a GitHub Actions cron).
The presets encode the strategy logic:

  * REVIEW exits ~30 min after the open: arm stops early, fire time-exits at live
    liquid prices (not next-day gaps), act on overnight news.
  * WATCH intraday: entry triggers (pullbacks / breakouts) happen during the
    session.
  * SCREEN after the close: rank on FINAL daily bars, review tickets in the
    evening, place at the next open (disciplined end-of-day swing).
"""

from __future__ import annotations

import datetime as _dt

# job -> list of Eastern "HH:MM" times.
PRESETS: dict[str, dict[str, list[str]]] = {
    # Recommended: properly-timed three jobs.
    "eod_swing": {
        "review": ["10:00"],
        "watch": ["11:00", "13:00", "15:00"],
        "screen": ["17:00"],
    },
    # Hands-off: one pre-open combined run + a midday watch.
    "morning": {
        "daily": ["08:00"],            # review + screen on yesterday's close
        "watch": ["12:00"],
    },
    # Active: more frequent intraday watch checks.
    "active": {
        "review": ["10:00"],
        "watch": ["10:30", "12:00", "13:30", "15:00"],
        "screen": ["17:00"],
    },
}

PRESET_LABELS = {
    "eod_swing": "End-of-day swing (recommended): review 10:00, watch "
                 "11/13/15, screen 17:00 ET",
    "morning": "Morning only (hands-off): one combined run 08:00 ET + watch 12:00",
    "active": "Active: review 10:00, watch 10:30/12/13:30/15, screen 17:00 ET",
    "custom": "Custom: your own ET times (Settings)",
}


def resolve_schedule(cfg) -> dict[str, list[str]]:
    """The job -> [ET times] map for the configured preset (or custom times)."""
    preset = getattr(cfg, "schedule_preset", "eod_swing") or "eod_swing"
    if preset == "custom":
        def _split(s):
            return [t.strip() for t in str(s or "").split(",") if t.strip()]
        out = {}
        if _split(cfg.custom_review_et):
            out["review"] = _split(cfg.custom_review_et)
        if _split(cfg.custom_screen_et):
            out["screen"] = _split(cfg.custom_screen_et)
        if _split(cfg.custom_watch_et):
            out["watch"] = _split(cfg.custom_watch_et)
        return out
    return {job: list(times) for job, times in PRESETS.get(preset, PRESETS["eod_swing"]).items()}


def screen_universes(cfg) -> list[str]:
    """The universe(s) a scheduled screen should run, in order."""
    raw = getattr(cfg, "scheduled_screen_universes", "") or cfg.screen_index or "sp500"
    return [u.strip() for u in str(raw).split(",") if u.strip()] or ["sp500"]


def _eastern_offset_hours(today: _dt.date | None = None) -> int:
    """Eastern's current UTC offset in hours (−4 EDT in summer, −5 EST), via the
    standard library — DST-correct without external data."""
    try:
        from zoneinfo import ZoneInfo
        ref = _dt.datetime.combine(today or _dt.date.today(), _dt.time(12),
                                   tzinfo=ZoneInfo("America/New_York"))
        return int(ref.utcoffset().total_seconds() // 3600)
    except Exception:
        return -4                                  # EDT fallback


def et_to_local(hhmm: str, today: _dt.date | None = None) -> str:
    """An Eastern 'HH:MM' converted to the LOCAL machine time (for schtasks)."""
    try:
        from zoneinfo import ZoneInfo
        h, m = (int(x) for x in hhmm.split(":"))
        et = _dt.datetime.combine(today or _dt.date.today(), _dt.time(h, m),
                                  tzinfo=ZoneInfo("America/New_York"))
        loc = et.astimezone()                      # machine local tz
        return loc.strftime("%H:%M")
    except Exception:
        return hhmm


def et_to_utc(hhmm: str, today: _dt.date | None = None) -> str:
    """An Eastern 'HH:MM' converted to UTC 'HH:MM' (for a GitHub Actions cron)."""
    h, m = (int(x) for x in hhmm.split(":"))
    total = (h - _eastern_offset_hours(today)) * 60 + m
    total %= 24 * 60
    return f"{total // 60:02d}:{total % 60:02d}"
