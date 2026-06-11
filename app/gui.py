"""Tkinter desktop GUI: configure, save, and run the system with live output.

No third-party GUI dependency (Tkinter ships with Python) so the PyInstaller
bundle stays self-contained and cross-platform. The look is a custom flat theme
(built on ttk's "clam") so it is modern and consistent on Windows and macOS.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext, ttk

from app import APP_NAME, APP_VERSION
from app.config import SECRET_FIELDS, AppConfig
from app.runner import (check_alpaca, place_manual_order, run_momentum_trade,
                        run_portfolio_status, run_recommendations,
                        run_reddit_scan, run_screen, run_strategy_backtest)

# (field, label, kind)  kind: "secret" | "text" | "int" | "float" | "choice"
_FIELDS = [
    ("anthropic_api_key", "Anthropic API key (LLM agents)", "secret"),
    ("alpaca_key_id", "Alpaca key id (broker)", "secret"),
    ("alpaca_secret", "Alpaca secret (broker)", "secret"),
    ("alpaca_env", "Alpaca environment", "choice"),
    ("edgar_user_agent", "EDGAR User-Agent (e.g. you@email.com)", "text"),
    ("reddit_client_id", "Reddit client id", "secret"),
    ("reddit_client_secret", "Reddit client secret", "secret"),
    ("reddit_username", "Reddit username (script app)", "text"),
    ("reddit_password", "Reddit password (script app)", "secret"),
    ("data_source", "Data source", "choice"),
    ("ticker", "Analyze one ticker (blank = scan universe)", "text"),
    ("n_symbols", "Universe size (symbols)", "int"),
    ("start_date", "Start date (YYYY-MM-DD)", "text"),
    ("end_date", "End date (YYYY-MM-DD)", "text"),
    ("seed", "Random seed", "int"),
    ("starting_equity", "Starting equity ($)", "float"),
    ("oos_start", "Out-of-sample start (YYYY-MM-DD)", "text"),
    ("insider_history_quarters", "Insider history (quarters)", "int"),
    ("filing_history_count", "Filing history (count)", "int"),
    ("momentum_hold_days", "Momentum hold (trading days)", "int"),
    ("momentum_max_positions", "Momentum max positions", "int"),
    ("reddit_top_k", "Reddit: tickers to analyze", "int"),
    ("screen_top_k", "Screen: deep-dive top K", "int"),
    ("screen_universe", "Screen: cap universe (0=all)", "int"),
    ("screen_index", "Screen index (default)", "choice"),
]

# Allowed values for "choice" fields.
_CHOICES = {"data_source": ["synthetic", "live"], "alpaca_env": ["paper", "live"],
            "screen_index": ["sp500", "qqq", "sp400", "sp600", "midsmall", "broad"]}

_FIELD_BY_NAME = {f[0]: f for f in _FIELDS}

# Config fields grouped into logical cards (purely presentational).
_GROUPS = [
    ("API keys & credentials",
     ["anthropic_api_key", "alpaca_key_id", "alpaca_secret", "alpaca_env",
      "edgar_user_agent", "reddit_client_id", "reddit_client_secret",
      "reddit_username", "reddit_password"]),
    ("Data & universe",
     ["data_source", "ticker", "n_symbols", "start_date", "end_date", "seed",
      "starting_equity", "oos_start"]),
    ("Strategy parameters",
     ["insider_history_quarters", "filing_history_count", "momentum_hold_days",
      "momentum_max_positions", "reddit_top_k", "screen_top_k", "screen_universe",
      "screen_index"]),
]

# -- palette (dark, layered: sidebar < content < card < raised; one accent) ----
SIDEBAR = "#0d1014"     # deepest (left rail)
BG = "#14181e"          # content background
CARD = "#1c212b"        # card / surface (raised above content)
SURF2 = "#262c39"       # raised controls (buttons, fields)
SURF3 = "#2f3744"       # hover
INK = "#eaedf3"         # primary text (off-white)
MUTED = "#8590a3"       # secondary text
FAINT = "#5a6577"       # tertiary / disabled
ACCENT = "#46c08a"      # emerald accent (used sparingly)
ACCENT_DK = "#37a074"
ACCENT_SOFT = "#1f3a30"  # accent-tinted surface (active nav)
ACCENT_INK = "#08120d"  # text on the accent
OK = "#5fd39a"
DANGER = "#e06c6c"
DANGER_SOFT = "#3a2026"
GEM = "#62c8d8"          # hidden-gem cyan
BORDER = "#272d38"
HOVER = "#161b22"        # sidebar hover
CONSOLE_BG = "#0b0e12"  # near-black console
CONSOLE_FG = "#c8cfdb"


class _NavItem(tk.Frame):
    """Sidebar navigation row: left accent indicator bar, hover state, click."""

    def __init__(self, parent, text: str, font, command):
        super().__init__(parent, bg=SIDEBAR)
        self.bar = tk.Frame(self, width=3, bg=SIDEBAR)
        self.bar.pack(side="left", fill="y")
        self.lbl = tk.Label(self, text=text, bg=SIDEBAR, fg=MUTED, anchor="w",
                            padx=15, pady=10, font=font, cursor="hand2")
        self.lbl.pack(side="left", fill="x", expand=True)
        self.active = False
        for w in (self, self.lbl):
            w.bind("<Button-1>", lambda e: command())
            w.bind("<Enter>", lambda e: self._hover(True))
            w.bind("<Leave>", lambda e: self._hover(False))

    def _hover(self, on: bool):
        if self.active:
            return
        bg = HOVER if on else SIDEBAR
        self.configure(bg=bg)
        self.lbl.configure(bg=bg, fg=INK if on else MUTED)
        self.bar.configure(bg=bg)

    def set_active(self, on: bool):
        self.active = on
        bg = HOVER if on else SIDEBAR
        self.configure(bg=bg)
        self.lbl.configure(bg=bg, fg=INK if on else MUTED)
        self.bar.configure(bg=ACCENT if on else bg)


class _ScrollFrame(ttk.Frame):
    """A vertically scrollable container; add children to ``.body``."""

    def __init__(self, parent):
        super().__init__(parent)
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self._canvas)
        self._win = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>",
                       lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfigure(self._win, width=e.width))
        self._canvas.bind("<Enter>", lambda e: self._bind_wheel())
        self._canvas.bind("<Leave>", lambda e: self._unbind_wheel())

    def _on_wheel(self, event):
        d = event.delta
        step = int(-d / 120) if abs(d) >= 120 else (-1 if d > 0 else 1)
        self._canvas.yview_scroll(step, "units")

    def _bind_wheel(self):
        self._canvas.bind_all("<MouseWheel>", self._on_wheel)
        self._canvas.bind_all("<Button-4>", lambda e: self._canvas.yview_scroll(-1, "units"))
        self._canvas.bind_all("<Button-5>", lambda e: self._canvas.yview_scroll(1, "units"))

    def _unbind_wheel(self):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")


class SwingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = AppConfig.load()
        self.vars: dict[str, tk.Variable] = {}
        self.q: queue.Queue = queue.Queue()
        self.running = False

        root.title(f"{APP_NAME} {APP_VERSION}")
        root.geometry("1120x860")
        root.minsize(940, 680)
        root.configure(bg=BG)
        self._setup_style()
        self._build()
        self.root.after(80, self._drain_queue)

    # -- theme -------------------------------------------------------------
    def _setup_style(self):
        fam = ("Segoe UI" if sys.platform == "win32"
               else "Helvetica Neue" if sys.platform == "darwin" else "DejaVu Sans")
        mono = ("Consolas" if sys.platform == "win32"
                else "Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono")
        self.f_base = tkfont.Font(family=fam, size=10)
        self.f_bold = tkfont.Font(family=fam, size=10, weight="bold")
        self.f_section = tkfont.Font(family=fam, size=9, weight="bold")
        self.f_title = tkfont.Font(family=fam, size=15, weight="bold")
        self.f_h1 = tkfont.Font(family=fam, size=19, weight="bold")
        self.f_sub = tkfont.Font(family=fam, size=9)
        self.f_nav = tkfont.Font(family=fam, size=10)
        self.f_mono = tkfont.Font(family=mono, size=10)

        # Dark-themed dropdown lists (tk Listbox inside ttk Combobox).
        self.root.option_add("*TCombobox*Listbox.background", CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", INK)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", ACCENT_INK)

        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=BG, foreground=INK, font=self.f_base,
                    fieldbackground=SURF2, bordercolor=BORDER)
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("Side.TFrame", background=SIDEBAR)
        s.configure("Rule.TFrame", background=BORDER)
        s.configure("AccentRule.TFrame", background=ACCENT)
        s.configure("TLabel", background=BG, foreground=INK)
        s.configure("Card.TLabel", background=CARD, foreground=INK)
        s.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=self.f_sub)
        s.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=self.f_sub)
        s.configure("Side.TLabel", background=SIDEBAR, foreground=INK)
        s.configure("SideMuted.TLabel", background=SIDEBAR, foreground=FAINT, font=self.f_sub)
        s.configure("Brand.TLabel", background=SIDEBAR, foreground=INK, font=self.f_title)
        s.configure("BrandSub.TLabel", background=SIDEBAR, foreground=ACCENT, font=self.f_sub)
        s.configure("PageTitle.TLabel", background=BG, foreground=INK, font=self.f_h1)
        s.configure("PageSub.TLabel", background=BG, foreground=MUTED, font=self.f_sub)
        s.configure("Status.TLabel", background=SIDEBAR, foreground=MUTED, font=self.f_sub)
        s.configure("OK.TLabel", background=BG, foreground=OK, font=self.f_sub)
        # Section headers that sit ABOVE cards (not inside a boxy LabelFrame).
        s.configure("Section.TLabel", background=BG, foreground=MUTED,
                    font=self.f_section)
        s.configure("SectionSub.TLabel", background=BG, foreground=FAINT,
                    font=self.f_sub)

        s.configure("TButton", background=SURF2, foreground=INK, bordercolor=SURF2,
                    relief="flat", padding=(14, 9), font=self.f_base, focuscolor=SURF2)
        s.map("TButton",
              background=[("active", SURF3), ("disabled", "#171b22")],
              foreground=[("disabled", FAINT)],
              bordercolor=[("active", SURF3)])
        s.configure("Accent.TButton", background=ACCENT, foreground=ACCENT_INK,
                    bordercolor=ACCENT, relief="flat", padding=(18, 9), font=self.f_bold,
                    focuscolor=ACCENT)
        s.map("Accent.TButton",
              background=[("active", ACCENT_DK), ("disabled", "#27483a")],
              foreground=[("disabled", "#7f8b85")])

        s.configure("TCheckbutton", background=CARD, foreground=INK, font=self.f_base)
        s.map("TCheckbutton", background=[("active", CARD)],
              foreground=[("disabled", MUTED)],
              indicatorcolor=[("selected", ACCENT), ("!selected", SURF2)])
        s.configure("TEntry", fieldbackground=SURF2, foreground=INK, bordercolor=BORDER,
                    insertcolor=INK, relief="flat", padding=6)
        s.map("TEntry", bordercolor=[("focus", ACCENT)])
        s.configure("TCombobox", fieldbackground=SURF2, foreground=INK, bordercolor=BORDER,
                    arrowcolor=MUTED, padding=5)
        s.map("TCombobox", fieldbackground=[("readonly", SURF2)],
              foreground=[("readonly", INK)], bordercolor=[("focus", ACCENT)])

        s.configure("TLabelframe", background=CARD, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
        s.configure("TLabelframe.Label", background=CARD, foreground=ACCENT,
                    font=self.f_section)
        s.configure("Vertical.TScrollbar", background=SURF2, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED, relief="flat", width=12)
        s.map("Vertical.TScrollbar", background=[("active", SURF3)])
        # Small toolbar buttons (console header), environment badges, busy strip.
        s.configure("Tool.TButton", background=CARD, foreground=MUTED,
                    bordercolor=BORDER, relief="flat", padding=(9, 4), font=self.f_sub)
        s.map("Tool.TButton", background=[("active", SURF2)],
              foreground=[("active", INK)])
        s.configure("BadgePaper.TLabel", background=ACCENT_SOFT, foreground=OK,
                    font=self.f_section, padding=(8, 2))
        s.configure("BadgeLive.TLabel", background=DANGER_SOFT, foreground=DANGER,
                    font=self.f_section, padding=(8, 2))
        s.configure("Accent.Horizontal.TProgressbar", background=ACCENT,
                    troughcolor=CARD, bordercolor=CARD,
                    lightcolor=ACCENT, darkcolor=ACCENT, thickness=3)

    # -- layout ------------------------------------------------------------
    def _build(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        # ---- left sidebar (brand + nav + status) ----
        side = ttk.Frame(outer, style="Side.TFrame", width=220)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        brand = ttk.Frame(side, style="Side.TFrame")
        brand.pack(fill="x", padx=20, pady=(24, 4))
        row = ttk.Frame(brand, style="Side.TFrame")
        row.pack(anchor="w")
        tk.Label(row, text="◆", bg=SIDEBAR, fg=ACCENT,
                 font=self.f_title).pack(side="left", padx=(0, 7))
        ttk.Label(row, text=APP_NAME, style="Brand.TLabel").pack(side="left")
        ttk.Label(brand, text="AI swing desk", style="BrandSub.TLabel").pack(
            anchor="w", padx=(1, 0))

        self._nav_btns: dict[str, _NavItem] = {}
        navwrap = ttk.Frame(side, style="Side.TFrame")
        navwrap.pack(fill="x", pady=(22, 0))
        for key, label in (("run", "Desk"), ("lessons", "Learning"),
                           ("settings", "Settings")):
            item = _NavItem(navwrap, label, self.f_nav, lambda k=key: self._show(k))
            item.pack(fill="x")
            self._nav_btns[key] = item

        statusbox = ttk.Frame(side, style="Side.TFrame")
        statusbox.pack(side="bottom", fill="x", padx=18, pady=16)
        self.env_badge = ttk.Label(statusbox, text="PAPER", style="BadgePaper.TLabel")
        self.env_badge.pack(anchor="w", pady=(0, 8))
        self.spinner = ttk.Label(statusbox, text="●  ready", style="Status.TLabel")
        self.spinner.pack(anchor="w")
        ttk.Label(statusbox, text=f"v{APP_VERSION}", style="SideMuted.TLabel").pack(
            anchor="w", pady=(4, 0))
        self._busy_task = ""
        self._busy_t0 = 0.0
        self._spin_i = 0
        self._refresh_env_badge()

        ttk.Frame(outer, style="Rule.TFrame", width=1).pack(side="left", fill="y")

        # ---- content area with stacked pages ----
        self._content = ttk.Frame(outer)
        self._content.pack(side="left", fill="both", expand=True)
        self._pages: dict[str, ttk.Frame] = {}
        for key, title, sub in (
            ("run", "Desk", "Screen the market, analyze a name, run the agents"),
            ("lessons", "Learning", "What the desk has learned from closed trades"),
            ("settings", "Settings", "Credentials, data, strategy parameters")):
            page = ttk.Frame(self._content)
            head = ttk.Frame(page)
            head.pack(fill="x", padx=26, pady=(22, 0))
            ttk.Label(head, text=title, style="PageTitle.TLabel").pack(anchor="w")
            ttk.Label(head, text=sub, style="PageSub.TLabel").pack(anchor="w", pady=(3, 0))
            body = ttk.Frame(page)
            body.pack(fill="both", expand=True)
            self._pages[key] = page
            page._body = body

        self._build_run(self._pages["run"]._body)
        self._build_lessons(self._pages["lessons"]._body)
        self._build_config(self._pages["settings"]._body)
        self._show("run")

    def _show(self, key: str):
        for k, page in self._pages.items():
            page.pack_forget()
        self._pages[key].pack(fill="both", expand=True)
        for k, item in self._nav_btns.items():
            item.set_active(k == key)
        if key == "lessons":
            self._refresh_lessons()

    # -- shared building blocks ---------------------------------------------
    def _card(self, parent, title: str | None = None, subtitle: str | None = None,
              expand: bool = False):
        """A flat, layered content card with its section header ABOVE it —
        the borderless-card look that replaces boxy labeled frames."""
        wrap = ttk.Frame(parent)
        wrap.pack(fill="both" if expand else "x", expand=expand,
                  padx=26, pady=(16, 0))
        if title:
            head = ttk.Frame(wrap)
            head.pack(fill="x", pady=(0, 7))
            ttk.Label(head, text=title.upper(), style="Section.TLabel").pack(side="left")
            if subtitle:
                ttk.Label(head, text=subtitle, style="SectionSub.TLabel").pack(
                    side="left", padx=(10, 0))
        card = tk.Frame(wrap, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1, bd=0)
        card.pack(fill="both" if expand else "x", expand=expand)
        inner = ttk.Frame(card, style="Card.TFrame")
        inner.pack(fill="both", expand=True, padx=16, pady=13)
        return inner

    def _mkbtn(self, parent, text, command, accent=False, width=None):
        b = ttk.Button(parent, text=text, command=command, cursor="hand2",
                       takefocus=False,
                       style="Accent.TButton" if accent else "TButton")
        if width:
            b.configure(width=width)
        self._action_buttons.append(b)
        return b

    def _add_fields(self, card, names):
        for i, name in enumerate(names):
            field, label, kind = _FIELD_BY_NAME[name]
            ttk.Label(card, text=label, style="Card.TLabel").grid(
                row=i, column=0, sticky="w", pady=5, padx=(4, 14))
            cur = getattr(self.cfg, field)
            var = tk.StringVar(value=str(cur))
            if kind == "choice":
                w = ttk.Combobox(card, textvariable=var, values=_CHOICES.get(field, []),
                                 state="readonly", width=34)
            else:
                show = "•" if kind == "secret" else ""
                w = ttk.Entry(card, textvariable=var, width=38, show=show)
            w.grid(row=i, column=1, sticky="ew", pady=5, padx=(0, 4))
            self.vars[field] = var
        card.columnconfigure(1, weight=1)

    def _build_config(self, parent):
        scroll = _ScrollFrame(parent)
        scroll.pack(fill="both", expand=True, padx=4, pady=4)
        body = scroll.body

        for title, names in _GROUPS:
            card = self._card(body, title)
            self._add_fields(card, names)

        opts = self._card(body, "Options")
        self.vars["use_llm_agents"] = tk.BooleanVar(value=self.cfg.use_llm_agents)
        ttk.Checkbutton(opts, text="Use LLM agents (experimental — may spend tokens)",
                        variable=self.vars["use_llm_agents"]).pack(anchor="w", padx=8, pady=2)
        self.vars["only_validated_edges"] = tk.BooleanVar(value=self.cfg.only_validated_edges)
        ttk.Checkbutton(opts, text="Only trade validated edges (run validation first)",
                        variable=self.vars["only_validated_edges"]).pack(anchor="w", padx=8, pady=2)
        self.vars["verbose_agents"] = tk.BooleanVar(value=self.cfg.verbose_agents)
        ttk.Checkbutton(opts, text="Show full agent reasoning (prompts, inputs, outputs)",
                        variable=self.vars["verbose_agents"]).pack(anchor="w", padx=8, pady=2)
        self.vars["place_orders"] = tk.BooleanVar(value=self.cfg.place_orders)
        ttk.Checkbutton(opts, text="Place approved orders on Alpaca (live deliberation)",
                        variable=self.vars["place_orders"]).pack(anchor="w", padx=8, pady=2)
        self.vars["learn_from_runs"] = tk.BooleanVar(value=self.cfg.learn_from_runs)
        ttk.Checkbutton(opts, text="Learn from runs (reflect on closed trades + recall lessons)",
                        variable=self.vars["learn_from_runs"]).pack(anchor="w", padx=8, pady=2)
        self.vars["auto_approve_lessons"] = tk.BooleanVar(value=self.cfg.auto_approve_lessons)
        ttk.Checkbutton(opts, text="Auto-activate new lessons (else review them on the Lessons tab)",
                        variable=self.vars["auto_approve_lessons"]).pack(anchor="w", padx=8, pady=2)
        self.vars["enable_live_trading"] = tk.BooleanVar(value=self.cfg.enable_live_trading)
        ttk.Checkbutton(opts, text="Enable LIVE (real-money) Alpaca env — extra gate",
                        variable=self.vars["enable_live_trading"],
                        command=self._warn_live).pack(anchor="w", padx=8, pady=2)
        ttk.Label(opts, style="Muted.TLabel", wraplength=820, justify="left",
                  text="Data source 'live' pulls REAL free data (Yahoo), cached locally; "
                       "'synthetic' is a planted-signal demo. Alpaca 'paper' = fake money, "
                       "'live' = REAL money (also needs the Enable-live gate). 'Place "
                       "approved orders' submits the live-deliberation's approved trades; "
                       "OFF = show proposals only. Keys are saved to "
                       "~/.swing_system/config.json (never committed or bundled).").pack(
            anchor="w", padx=8, pady=(6, 4))

        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=26, pady=14)
        ttk.Button(bar, text="Save configuration", style="Accent.TButton",
                   cursor="hand2", takefocus=False,
                   command=self._save).pack(side="left")
        self._mkbtn(bar, "Test Alpaca connection",
                    lambda: self._start(check_alpaca)).pack(side="left", padx=(10, 0))
        self.status = ttk.Label(bar, text="", style="OK.TLabel")
        self.status.pack(side="left", padx=14)

    # Screen universes shown in the picker (label -> screen_index).
    _SCREENS = {
        "S&P 500 — large caps": "sp500",
        "Nasdaq-100 (QQQ)": "qqq",
        "S&P 400 — mid caps": "sp400",
        "S&P 600 — small caps": "sp600",
        "Mid + small caps — hidden gems": "midsmall",
        "S&P 1500 — broad sweep": "broad",
    }

    def _build_run(self, parent):
        self._action_buttons: list[ttk.Button] = []

        # ---- find opportunities: one screen control + single-name analysis ----
        g = self._card(parent, "Find opportunities",
                       "free pre-filter over the whole index, AI agent panel on the best few")
        ttk.Label(g, text="Screen", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        self._screen_choice = tk.StringVar(value=next(iter(self._SCREENS)))
        ttk.Combobox(g, textvariable=self._screen_choice, values=list(self._SCREENS),
                     state="readonly", width=30).grid(row=0, column=1, sticky="w")
        self._mkbtn(g, "Run screen  ▸", self._run_screen_choice, accent=True).grid(
            row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Label(g, text="Analyze", style="Card.TLabel").grid(
            row=0, column=3, sticky="e", padx=(36, 12))
        self.ent_ticker = ttk.Entry(g, width=12)
        self.ent_ticker.insert(0, self.cfg.ticker or "")
        self.ent_ticker.grid(row=0, column=4, sticky="w")
        self.ent_ticker.bind("<Return>", lambda e: self._analyze_ticker())
        self._mkbtn(g, "Deep-dive  ▸", self._analyze_ticker, accent=True).grid(
            row=0, column=5, sticky="w", padx=(12, 0))
        g.columnconfigure(3, weight=1)

        # ---- trade & monitor ----
        rows = self._card(parent, "Trade & monitor")
        actions = [
            ("Momentum trade", lambda: self._start(run_momentum_trade)),
            ("Portfolio P&L", lambda: self._start(run_portfolio_status)),
            ("Strategy backtest", lambda: self._start(run_strategy_backtest)),
            ("Reddit sentiment", lambda: self._start(run_reddit_scan)),
        ]
        for i, (label, cmd) in enumerate(actions):
            self._mkbtn(rows, label, cmd).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            rows.columnconfigure(i, weight=1, uniform="desk")

        # ---- selective execution: tickets from the last run ----
        self.orders_body = self._card(parent, "Order tickets",
                                      "BUY recommendations from the last run")
        self._orders_hint = ttk.Label(
            self.orders_body, style="CardMuted.TLabel",
            text="Run a screen or a deep-dive — BUY recommendations appear here as "
                 "tickets you can review and place individually.")
        self._orders_hint.pack(anchor="w")

        # ---- output console with its own toolbar ----
        cons = self._card(parent, "Output", expand=True)
        tools = ttk.Frame(cons, style="Card.TFrame")
        tools.pack(fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(tools, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar", length=180)
        ttk.Button(tools, text="Open logs folder", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._open_logs).pack(side="right", padx=(6, 0))
        ttk.Button(tools, text="Copy output", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._copy_output).pack(side="right", padx=(6, 0))
        ttk.Button(tools, text="Clear", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._clear).pack(side="right")
        self.out = scrolledtext.ScrolledText(
            cons, wrap="word", height=20, font=self.f_mono, bg=CONSOLE_BG,
            fg=CONSOLE_FG, insertbackground=CONSOLE_FG, relief="flat", bd=0,
            padx=14, pady=12)
        self.out.pack(fill="both", expand=True)
        self.out.tag_configure("err", foreground=DANGER)
        self.out.tag_configure("warn", foreground="#d8a657")
        self.out.tag_configure("ok", foreground="#57c98a")
        self.out.tag_configure("hl", foreground="#7fa8e8")
        self.out.tag_configure("gem", foreground=GEM)
        self.out.tag_configure("dim", foreground="#6f7b8e")
        self.out.configure(state="disabled")
        ttk.Frame(parent).pack(pady=5)            # bottom breathing room
        self._log(f"{APP_NAME} ready. Pick a universe and run a screen, or deep-dive "
                  "a single ticker. Credentials live under Settings.")

    def _run_screen_choice(self):
        key = self._SCREENS.get(self._screen_choice.get(), "sp500")
        self._start(run_screen, screen_index=key)

    def _copy_output(self):
        text = self.out.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _build_lessons(self, parent):
        card = self._card(parent, "Learning memory",
                          "advisory lessons + base rates that inform future runs",
                          expand=True)
        bar = ttk.Frame(card, style="Card.TFrame")
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="Refresh", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._refresh_lessons).pack(side="right")
        ttk.Button(bar, text="Clear all", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._clear_lessons).pack(
            side="right", padx=(0, 6))
        ttk.Button(bar, text="Approve all", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._approve_lessons).pack(
            side="right", padx=(0, 6))
        self.lessons_out = scrolledtext.ScrolledText(
            card, wrap="word", font=self.f_mono, bg=CONSOLE_BG, fg=CONSOLE_FG,
            relief="flat", bd=0, padx=14, pady=12)
        self.lessons_out.pack(fill="both", expand=True)
        self.lessons_out.configure(state="disabled")
        ttk.Frame(parent).pack(pady=5)
        self._refresh_lessons()

    def _refresh_lessons(self):
        try:
            from app.learning import load_memory, summarize
            from app import reco_ledger
            text = (summarize(load_memory())
                    + "\n\n" + "-" * 60 + "\n"
                    + reco_ledger.summarize())
        except Exception as exc:
            text = f"(could not load learning memory: {exc})"
        self.lessons_out.configure(state="normal")
        self.lessons_out.delete("1.0", "end")
        self.lessons_out.insert("end", text + "\n")
        self.lessons_out.configure(state="disabled")

    def _approve_lessons(self):
        try:
            from app.learning import load_memory, save_memory
            mem = load_memory()
            for e in mem.entries:
                e.human_reviewed = True
            save_memory(mem)
        except Exception as exc:
            messagebox.showerror("Approve failed", str(exc))
            return
        self._refresh_lessons()

    def _clear_lessons(self):
        if not messagebox.askyesno("Clear learning memory",
                                   "Delete all accumulated lessons and trade outcomes? "
                                   "This cannot be undone."):
            return
        try:
            from app.learning import LEARNING_PATH
            if LEARNING_PATH.exists():
                LEARNING_PATH.unlink()
        except Exception as exc:
            messagebox.showerror("Clear failed", str(exc))
            return
        self._refresh_lessons()

    # -- actions -----------------------------------------------------------
    def _collect(self) -> AppConfig:
        d = {}
        for field, _, kind in _FIELDS:
            val = self.vars[field].get()
            if kind == "int":
                val = int(float(val))
            elif kind == "float":
                val = float(val)
            d[field] = val
        d["use_llm_agents"] = bool(self.vars["use_llm_agents"].get())
        d["only_validated_edges"] = bool(self.vars["only_validated_edges"].get())
        d["verbose_agents"] = bool(self.vars["verbose_agents"].get())
        d["place_orders"] = bool(self.vars["place_orders"].get())
        d["enable_live_trading"] = bool(self.vars["enable_live_trading"].get())
        d["learn_from_runs"] = bool(self.vars["learn_from_runs"].get())
        d["auto_approve_lessons"] = bool(self.vars["auto_approve_lessons"].get())
        return AppConfig(**d)

    # -- selective execution ----------------------------------------------
    def _show_orders(self, result: dict):
        """Populate the orders panel with the last run's BUY recommendations."""
        for w in self.orders_body.winfo_children():
            w.destroy()
        recs = result.get("recommendations", [])
        port = {p["symbol"]: p for p in result.get("portfolio", [])}
        if not recs:
            ttk.Label(self.orders_body, style="Muted.TLabel",
                      text="Last run produced no BUY recommendations to place.").pack(
                anchor="w", padx=4, pady=4)
            return
        env = self._collect().alpaca_env.upper()
        ttk.Label(self.orders_body, style="CardMuted.TLabel",
                  text=f"{len(recs)} ticket(s) — review qty / type / price, then Place. "
                       f"Orders route to your {env} Alpaca account.").pack(
            anchor="w", padx=2, pady=(0, 6))
        grid = ttk.Frame(self.orders_body, style="Card.TFrame")
        grid.pack(fill="x")
        for c, head in enumerate(("symbol", "conv", "ref price", "qty", "type",
                                  "limit $", "bracket", "")):
            ttk.Label(grid, text=head, style="CardMuted.TLabel").grid(
                row=0, column=c, sticky="w", padx=(0, 10), pady=(0, 3))
        for i, r in enumerate(recs, start=1):
            sym = r["symbol"]
            entry = r.get("entry") or 0
            stop, target = r.get("stop"), r.get("target")
            default_qty = port.get(sym, {}).get("shares") or r.get("shares_at_ref_equity") or 0
            cell = ttk.Frame(grid, style="Card.TFrame")
            cell.grid(row=i, column=0, sticky="w", padx=(0, 10), pady=5)
            ttk.Label(cell, text=sym, style="Card.TLabel").pack(side="left")
            if r.get("hidden_gem"):
                tk.Label(cell, text=" ◆", bg=CARD, fg=GEM,
                         font=self.f_base).pack(side="left")
            ttk.Label(grid, text=f"{r.get('conviction', 0):.2f}",
                      style="CardMuted.TLabel").grid(row=i, column=1, sticky="w",
                                                     padx=(0, 10), pady=5)
            ttk.Label(grid, text=f"~${entry}", style="CardMuted.TLabel").grid(
                row=i, column=2, sticky="w", padx=(0, 10), pady=5)
            qv = tk.StringVar(value=str(int(default_qty)))
            ttk.Entry(grid, textvariable=qv, width=6).grid(
                row=i, column=3, sticky="w", padx=(0, 10), pady=5)
            tv = tk.StringVar(value="limit")
            ttk.Combobox(grid, textvariable=tv, values=["market", "limit"],
                         state="readonly", width=7).grid(
                row=i, column=4, sticky="w", padx=(0, 10), pady=5)
            # The PM's pullback entry (an 'adjust' decision) pre-fills the limit:
            # the most actionable number in the deliberation, not the last close.
            pv = tk.StringVar(value=str(r.get("suggested_entry") or entry))
            ttk.Entry(grid, textvariable=pv, width=9).grid(
                row=i, column=5, sticky="w", padx=(0, 10), pady=5)
            bv = tk.BooleanVar(value=bool(stop and target))
            ttk.Checkbutton(grid, text="stop/target", variable=bv).grid(
                row=i, column=6, sticky="w", padx=(0, 14), pady=5)
            ttk.Button(grid, text="Place order", style="Accent.TButton",
                       cursor="hand2", takefocus=False,
                       command=lambda r=r, qv=qv, tv=tv, pv=pv, bv=bv:
                       self._place_order(r, qv, tv, pv, bv)).grid(
                row=i, column=7, sticky="w", pady=5)
        notes = []
        if any(r.get("hidden_gem") for r in recs):
            notes.append("◆ = hidden-gem pick (early acceleration)")
        if any(r.get("suggested_entry") for r in recs):
            notes.append("limit pre-set to the PM's pullback entry where it advised "
                         "waiting for a dip (below the ref price)")
        if notes:
            ttk.Label(self.orders_body, text="   ·   ".join(notes),
                      style="CardMuted.TLabel").pack(anchor="w", padx=2, pady=(6, 0))

    def _place_order(self, rec, qv, tv, pv, bv):
        try:
            qty = int(float(qv.get()))
            otype = tv.get()
            price = float(pv.get()) if (otype == "limit" and pv.get()) else None
        except ValueError:
            messagebox.showerror("Invalid order", "Quantity and price must be numbers.")
            return
        if qty <= 0:
            messagebox.showerror("Invalid order", "Quantity must be greater than 0.")
            return
        cfg = self._collect()
        env = cfg.alpaca_env.upper()
        px = f"limit ${price}" if otype == "limit" else "market"
        if not messagebox.askyesno(
                "Confirm order",
                f"Submit {env} BUY {qty} {rec['symbol']} ({px})"
                f"{' with stop/target' if bv.get() else ''}?\n\n"
                f"This sends a real order to your {env} Alpaca account."):
            return
        order = {"symbol": rec["symbol"], "qty": qty, "order_type": otype,
                 "limit_price": price, "stop": rec.get("stop"),
                 "target": rec.get("target"), "attach_bracket": bool(bv.get()),
                 "ref_price": rec.get("entry")}
        self._log(f"\n[order] submitting BUY {qty} {rec['symbol']} ({px}) ...")
        threading.Thread(
            target=lambda: place_manual_order(
                cfg, order, lambda line: self.q.put(("log", line))),
            daemon=True).start()

    def _save(self):
        try:
            cfg = self._collect()
            path = cfg.save()
            self.cfg = cfg
            self.status.config(text=f"Saved to {path}")
            self._refresh_env_badge()
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _refresh_env_badge(self):
        """Always-visible trading-environment indicator (paper vs real money)."""
        try:
            env = str(self.vars["alpaca_env"].get() if "alpaca_env" in self.vars
                      else self.cfg.alpaca_env).lower()
        except Exception:
            env = "paper"
        if env == "live":
            self.env_badge.configure(text="⚠ LIVE — REAL MONEY", style="BadgeLive.TLabel")
        else:
            self.env_badge.configure(text="PAPER TRADING", style="BadgePaper.TLabel")

    def _warn_live(self):
        if self.vars["enable_live_trading"].get():
            messagebox.showwarning(
                "Real-money trading",
                "Enabling LIVE allows the app to place orders on your REAL-money Alpaca "
                "account (api.alpaca.markets) when Alpaca environment = 'live' AND "
                "'Place approved orders on Alpaca' is on. Real capital can be lost. "
                "Default to 'paper' until you have validated everything. Use the kill "
                "switch / your Alpaca dashboard to halt.")

    def _analyze_ticker(self):
        sym = self.ent_ticker.get().strip().upper()
        if not sym:
            messagebox.showinfo("Ticker required",
                                "Type a stock symbol (e.g. NVDA) to analyze.")
            return
        self._start(run_recommendations, ticker=sym)

    def _start(self, fn, **overrides):
        if self.running:
            return
        try:
            cfg = self._collect()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        for k, v in overrides.items():             # per-button config overrides
            setattr(cfg, k, v)
        self.running = True
        task = fn.__name__.replace("run_", "").replace("_", " ")
        self._show("run")                          # output lives on the Desk
        self._set_busy(True, task)
        self._log(f"\n=== starting: {fn.__name__} ===")

        def work():
            try:
                res = fn(cfg, lambda line: self.q.put(("log", line)))
                self.q.put(("result", res))
                self.q.put(("done", None))
            except Exception as exc:  # surface, never crash the GUI
                self.q.put(("log", f"[error] {type(exc).__name__}: {exc}"))
                self.q.put(("done", None))

        threading.Thread(target=work, daemon=True).start()

    # -- queue pump --------------------------------------------------------
    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "result":
                    if isinstance(payload, dict) and payload.get("recommendations"):
                        self._show_orders(payload)
                elif kind == "done":
                    self.running = False
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

    _SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _set_busy(self, busy: bool, task: str = ""):
        state = "disabled" if busy else "normal"
        for b in self._action_buttons:
            b.config(state=state)
        if busy:
            import time
            self._busy_task = task or "working"
            self._busy_t0 = time.time()
            self.progress.pack(side="left", padx=(4, 0), pady=2)
            self.progress.start(24)
            self._tick_spinner()
        else:
            self._busy_task = ""
            self.progress.stop()
            self.progress.pack_forget()
            self.spinner.config(text="●  ready")

    def _tick_spinner(self):
        """Animated sidebar status with the running task + elapsed time."""
        if not self._busy_task:
            return
        import time
        self._spin_i = (self._spin_i + 1) % len(self._SPIN)
        secs = int(time.time() - self._busy_t0)
        self.spinner.config(
            text=f"{self._SPIN[self._spin_i]}  {self._busy_task} · {secs // 60}:{secs % 60:02d}")
        self.root.after(160, self._tick_spinner)

    @staticmethod
    def _tag_for(line: str) -> str | None:
        s = line.strip()
        low = s.lower()
        if (s.startswith("!!") or "[error]" in low or "[blocked]" in low
                or "verdict: do not buy" in low):
            return "err"
        if "[warn" in low or "warning" in low or low.startswith("[note]") \
                or "[hint]" in low:
            return "warn"
        if "hidden gem" in low or "hidden-gem" in low:
            return "gem"
        if ("verdict: buy" in low or "recommend buy" in low or s.startswith("[done]")
                or "[selftest] ok" in low):
            return "ok"
        if s.startswith("===") or s.startswith("ANALYSIS:") or ">>> AGENT" in s \
                or s.startswith("####") or "ANALYST" in s.upper()[:14]:
            return "hl"
        if s.startswith("[") or s.startswith("  -") or s.startswith("  +"):
            return "dim"
        return None

    def _log(self, line: str):
        self.out.configure(state="normal")
        tag = self._tag_for(line)
        self.out.insert("end", line + "\n", tag or ())
        self.out.see("end")
        self.out.configure(state="disabled")

    def _clear(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")

    def _open_logs(self):
        import os
        import subprocess
        from app.runner import LOGS_DIR
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(LOGS_DIR)            # noqa: SIM
            elif sys.platform == "darwin":
                subprocess.run(["open", str(LOGS_DIR)])
            else:
                subprocess.run(["xdg-open", str(LOGS_DIR)])
        except Exception as exc:
            messagebox.showinfo("Logs folder", f"Logs are in:\n{LOGS_DIR}\n\n({exc})")


def launch():
    root = tk.Tk()
    SwingApp(root)
    root.mainloop()
