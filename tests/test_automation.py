"""Scheduling: presets, ET timezone conversion, multi-universe, env-overlay config."""

import datetime as dt

from app import schedule as sch
from app.config import AppConfig


def test_presets_cover_the_three_jobs():
    s = sch.resolve_schedule(AppConfig(schedule_preset="eod_swing"))
    assert s["review"] == ["10:00"]
    assert s["watch"] == ["11:00", "13:00", "15:00"]
    assert s["screen"] == ["17:00"]
    # 'morning' is one combined daily run + a midday watch.
    m = sch.resolve_schedule(AppConfig(schedule_preset="morning"))
    assert m["daily"] == ["08:00"] and "screen" not in m


def test_custom_preset_reads_custom_times():
    cfg = AppConfig(schedule_preset="custom", custom_review_et="09:45",
                    custom_screen_et="16:30", custom_watch_et="11:30,14:30")
    s = sch.resolve_schedule(cfg)
    assert s["review"] == ["09:45"] and s["screen"] == ["16:30"]
    assert s["watch"] == ["11:30", "14:30"]


def test_screen_universes_is_a_comma_list():
    assert sch.screen_universes(AppConfig(scheduled_screen_universes="sp500,midsmall")) \
        == ["sp500", "midsmall"]
    # Falls back to the single configured index when the list is blank.
    assert sch.screen_universes(AppConfig(scheduled_screen_universes="",
                                          screen_index="qqq")) == ["qqq"]


def test_et_to_utc_handles_dst():
    # Summer (EDT = UTC-4): 10:00 ET -> 14:00 UTC.
    assert sch.et_to_utc("10:00", dt.date(2026, 7, 1)) == "14:00"
    # Winter (EST = UTC-5): 10:00 ET -> 15:00 UTC.
    assert sch.et_to_utc("10:00", dt.date(2026, 1, 15)) == "15:00"
    # 17:00 ET summer -> 21:00 UTC (the screen cron).
    assert sch.et_to_utc("17:00", dt.date(2026, 7, 1)) == "21:00"


def test_env_overlay_builds_config_without_a_file(tmp_path, monkeypatch):
    # No config file present: config comes entirely from the environment.
    monkeypatch.setattr("app.config.CONFIG_PATH", tmp_path / "nope.json")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-xyz")
    monkeypatch.setenv("ALPACA_ENV", "paper")
    monkeypatch.setenv("SWING_DATA_SOURCE", "live")
    monkeypatch.setenv("SWING_USE_LLM", "true")
    monkeypatch.setenv("SWING_PLACE_ORDERS", "false")
    monkeypatch.setenv("SWING_AUTO_MANAGE_EXITS", "true")
    monkeypatch.setenv("SWING_SCREEN_UNIVERSES", "sp500,midsmall")
    cfg = AppConfig.load()
    assert cfg.anthropic_api_key == "sk-xyz" and cfg.data_source == "live"
    assert cfg.use_llm_agents is True and cfg.place_orders is False
    assert cfg.auto_manage_exits is True
    assert cfg.scheduled_screen_universes == "sp500,midsmall"


def test_headless_screen_universes_parsing(monkeypatch):
    from app import main

    monkeypatch.setattr(AppConfig, "load",
                        classmethod(lambda cls: AppConfig(scheduled_screen_universes="sp400")))
    # Explicit arg wins, filtered to valid universes.
    assert main._screen_universes(["--screen", "sp500,midsmall,bogus"]) == ["sp500", "midsmall"]
    # No arg -> the configured default.
    assert main._screen_universes(["--screen"]) == ["sp400"]
