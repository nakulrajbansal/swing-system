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
from app.runner import (check_alpaca, place_manual_order, run_deliberation,
                        run_filing_validation, run_insider_validation,
                        run_momentum_trade, run_paper, run_portfolio_status,
                        run_recommendations, run_reddit_scan, run_screen,
                        run_strategy_backtest, run_validation)

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
    ("screen_top_k", "S&P 500 screen: deep-dive top K", "int"),
    ("screen_universe", "S&P 500 screen: cap universe (0=all)", "int"),
]

# Allowed values for "choice" fields.
_CHOICES = {"data_source": ["synthetic", "live"], "alpaca_env": ["paper", "live"]}

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
      "momentum_max_positions", "reddit_top_k", "screen_top_k", "screen_universe"]),
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
BORDER = "#272d38"
CONSOLE_BG = "#0b0e12"  # near-black console
CONSOLE_FG = "#c8cfdb"


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
        root.geometry("1000x820")
        root.minsize(820, 620)
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
        self.f_title = tkfont.Font(family=fam, size=14, weight="bold")
        self.f_h1 = tkfont.Font(family=fam, size=16, weight="bold")
        self.f_sub = tkfont.Font(family=fam, size=9)
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

        s.configure("TButton", background=SURF2, foreground=INK, bordercolor=BORDER,
                    relief="flat", padding=(13, 8), font=self.f_base, focuscolor=BG)
        s.map("TButton",
              background=[("active", SURF3), ("disabled", "#171b22")],
              foreground=[("disabled", FAINT)],
              bordercolor=[("active", ACCENT)])
        s.configure("Accent.TButton", background=ACCENT, foreground=ACCENT_INK,
                    bordercolor=ACCENT, relief="flat", padding=(16, 10), font=self.f_bold)
        s.map("Accent.TButton",
              background=[("active", ACCENT_DK), ("disabled", "#27483a")],
              foreground=[("disabled", "#7f8b85")])
        # Sidebar nav items (flat, left-aligned).
        s.configure("Nav.TButton", background=SIDEBAR, foreground=MUTED, bordercolor=SIDEBAR,
                    relief="flat", padding=(16, 11), font=self.f_base, anchor="w")
        s.map("Nav.TButton", background=[("active", "#161b22")],
              foreground=[("active", INK)])
        s.configure("NavActive.TButton", background=ACCENT_SOFT, foreground=ACCENT,
                    bordercolor=ACCENT_SOFT, relief="flat", padding=(16, 11),
                    font=self.f_bold, anchor="w")
        s.map("NavActive.TButton", background=[("active", ACCENT_SOFT)],
              foreground=[("active", ACCENT)])

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

    # -- layout ------------------------------------------------------------
    def _build(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        # ---- left sidebar (brand + nav + status) ----
        side = ttk.Frame(outer, style="Side.TFrame", width=208)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        brand = ttk.Frame(side, style="Side.TFrame")
        brand.pack(fill="x", padx=18, pady=(20, 6))
        ttk.Label(brand, text="◎ " + APP_NAME, style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand, text="AI swing desk", style="BrandSub.TLabel").pack(anchor="w")
        ttk.Frame(side, style="Rule.TFrame", height=1).pack(fill="x", padx=14, pady=(10, 8))

        self._nav_btns: dict[str, ttk.Button] = {}
        navwrap = ttk.Frame(side, style="Side.TFrame")
        navwrap.pack(fill="x", padx=10)
        for key, label in (("run", "  Desk"), ("lessons", "  Learning"),
                           ("settings", "  Settings")):
            b = ttk.Button(navwrap, text=label, style="Nav.TButton",
                           takefocus=False, command=lambda k=key: self._show(k))
            b.pack(fill="x", pady=2)
            self._nav_btns[key] = b

        statusbox = ttk.Frame(side, style="Side.TFrame")
        statusbox.pack(side="bottom", fill="x", padx=18, pady=16)
        self.spinner = ttk.Label(statusbox, text="● ready", style="Status.TLabel")
        self.spinner.pack(anchor="w")
        ttk.Label(statusbox, text=f"v{APP_VERSION}", style="SideMuted.TLabel").pack(anchor="w")

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
            head.pack(fill="x", padx=22, pady=(18, 0))
            ttk.Label(head, text=title, style="PageTitle.TLabel").pack(anchor="w")
            ttk.Label(head, text=sub, style="PageSub.TLabel").pack(anchor="w", pady=(2, 0))
            ttk.Frame(page, style="Rule.TFrame", height=1).pack(fill="x", padx=22, pady=(12, 0))
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
        for k, btn in self._nav_btns.items():
            btn.configure(style="NavActive.TButton" if k == key else "Nav.TButton")
        if key == "lessons":
            self._refresh_lessons()

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
            card = ttk.LabelFrame(body, text=f" {title} ")
            card.pack(fill="x", padx=10, pady=(8, 4), ipady=4)
            self._add_fields(card, names)

        opts = ttk.LabelFrame(body, text=" Options ")
        opts.pack(fill="x", padx=10, pady=(8, 4), ipady=4)
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
        bar.pack(fill="x", padx=14, pady=10)
        ttk.Button(bar, text="Save configuration", style="Accent.TButton",
                   command=self._save).pack(side="left")
        self.status = ttk.Label(bar, text="", style="OK.TLabel")
        self.status.pack(side="left", padx=14)

    def _build_run(self, parent):
        # Single-ticker analysis.
        card0 = ttk.LabelFrame(parent, text=" Analyze a single stock ")
        card0.pack(fill="x", padx=14, pady=(12, 6), ipady=6)
        bar0 = ttk.Frame(card0, style="Card.TFrame")
        bar0.pack(fill="x", padx=8, pady=6)
        ttk.Label(bar0, text="Ticker:", style="Card.TLabel").pack(side="left")
        self.ent_ticker = ttk.Entry(bar0, width=12)
        self.ent_ticker.insert(0, self.cfg.ticker or "")
        self.ent_ticker.pack(side="left", padx=(6, 8))
        self.btn_ticker = ttk.Button(bar0, text="Analyze this ticker", style="Accent.TButton",
                                     command=lambda: self._analyze_ticker())
        self.btn_ticker.pack(side="left")
        ttk.Label(bar0, text="  runs the full AI-agent panel on just this name",
                  style="Muted.TLabel").pack(side="left")

        # Primary actions.
        card1 = ttk.LabelFrame(parent, text=" Find opportunities ")
        card1.pack(fill="x", padx=14, pady=6, ipady=6)
        bar = ttk.Frame(card1, style="Card.TFrame")
        bar.pack(fill="x", padx=8, pady=6)
        self.btn_screen = ttk.Button(bar, text="◎  Screen S&P 500 (find best)",
                                     style="Accent.TButton",
                                     command=lambda: self._start(run_screen))
        self.btn_screen.pack(side="left", padx=(0, 8))
        self.btn_recs = ttk.Button(bar, text="Scan core universe",
                                   command=lambda: self._start(run_recommendations, ticker=""))
        self.btn_recs.pack(side="left", padx=(0, 8))
        self.btn_momentum = ttk.Button(bar, text="Momentum trade (enter/exit)",
                                       command=lambda: self._start(run_momentum_trade))
        self.btn_momentum.pack(side="left", padx=(0, 8))
        self.btn_reddit = ttk.Button(bar, text="Reddit scan",
                                     command=lambda: self._start(run_reddit_scan))
        self.btn_reddit.pack(side="left", padx=(0, 8))
        self.btn_delib = ttk.Button(bar, text="Live deliberation (gated)",
                                    command=lambda: self._start(run_deliberation))
        self.btn_delib.pack(side="left", padx=(0, 8))

        # Tools / validation.
        card2 = ttk.LabelFrame(parent, text=" Validation & tools ")
        card2.pack(fill="x", padx=14, pady=6, ipady=6)
        bar2 = ttk.Frame(card2, style="Card.TFrame")
        bar2.pack(fill="x", padx=8, pady=6)
        self.btn_val = ttk.Button(bar2, text="Validation harness",
                                  command=lambda: self._start(run_validation))
        self.btn_val.pack(side="left", padx=(0, 8))
        self.btn_backtest = ttk.Button(bar2, text="Strategy backtest (vs S&P 500)",
                                       command=lambda: self._start(run_strategy_backtest))
        self.btn_backtest.pack(side="left", padx=(0, 8))
        self.btn_paper = ttk.Button(bar2, text="Paper backtest",
                                    command=lambda: self._start(run_paper))
        self.btn_paper.pack(side="left", padx=(0, 8))
        self.btn_hist = ttk.Button(bar2, text="Validate history: insider",
                                   command=lambda: self._start(run_insider_validation))
        self.btn_hist.pack(side="left", padx=(0, 8))
        self.btn_filings = ttk.Button(bar2, text="Validate history: filings",
                                      command=lambda: self._start(run_filing_validation))
        self.btn_filings.pack(side="left", padx=(0, 8))
        self.btn_portfolio = ttk.Button(bar2, text="Paper portfolio (P&L)",
                                        command=lambda: self._start(run_portfolio_status))
        self.btn_portfolio.pack(side="left", padx=(0, 8))
        self.btn_alpaca = ttk.Button(bar2, text="Check Alpaca",
                                     command=lambda: self._start(check_alpaca))
        self.btn_alpaca.pack(side="left", padx=(0, 8))
        ttk.Button(bar2, text="Clear", command=self._clear).pack(side="left", padx=(0, 8))
        ttk.Button(bar2, text="Open logs", command=self._open_logs).pack(side="left")

        # Selective execution: populated from the last run's BUY recommendations.
        self.orders_card = ttk.LabelFrame(parent, text=" Place orders (from last run) ")
        self.orders_card.pack(fill="x", padx=14, pady=6)
        self.orders_body = ttk.Frame(self.orders_card, style="Card.TFrame")
        self.orders_body.pack(fill="x", padx=8, pady=6)
        self._orders_hint = ttk.Label(
            self.orders_body, style="Muted.TLabel",
            text="Run 'Screen S&P 500' or analyze a ticker — buyable tickets appear "
                 "here with qty / price / order-type and a Place button.")
        self._orders_hint.pack(anchor="w", padx=4, pady=4)

        # Output console.
        card3 = ttk.LabelFrame(parent, text=" Output ")
        card3.pack(fill="both", expand=True, padx=14, pady=(6, 6))
        self.out = scrolledtext.ScrolledText(
            card3, wrap="word", height=20, font=self.f_mono, bg=CONSOLE_BG,
            fg=CONSOLE_FG, insertbackground=CONSOLE_FG, relief="flat", bd=0,
            padx=12, pady=10)
        self.out.pack(fill="both", expand=True, padx=4, pady=4)
        self.out.tag_configure("err", foreground="#e06c6c")
        self.out.tag_configure("warn", foreground="#d8a657")
        self.out.tag_configure("ok", foreground="#57c98a")
        self.out.tag_configure("hl", foreground="#7fa8e8", font=self.f_mono)
        self.out.tag_configure("dim", foreground="#6f7b8e")
        self.out.configure(state="disabled")
        self._log(f"{APP_NAME} ready. Screen the S&P 500 to find the best names, or "
                  "analyze a ticker. Set keys under Settings.")

    def _build_lessons(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=22, pady=(14, 4))
        ttk.Label(bar, text="Advisory lessons + base rates that inform the agents "
                  "on future runs.", style="PageSub.TLabel").pack(side="left")
        ttk.Button(bar, text="Refresh", command=self._refresh_lessons).pack(side="right")
        ttk.Button(bar, text="Approve all", style="Accent.TButton",
                   command=self._approve_lessons).pack(side="right", padx=(0, 8))
        ttk.Button(bar, text="Clear all", command=self._clear_lessons).pack(side="right", padx=(0, 8))

        card = ttk.LabelFrame(parent, text=" Learning memory ")
        card.pack(fill="both", expand=True, padx=14, pady=(4, 12))
        self.lessons_out = scrolledtext.ScrolledText(
            card, wrap="word", font=self.f_mono, bg=CONSOLE_BG, fg=CONSOLE_FG,
            relief="flat", bd=0, padx=12, pady=10)
        self.lessons_out.pack(fill="both", expand=True, padx=4, pady=4)
        self.lessons_out.configure(state="disabled")
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
                  text=f"{len(recs)} ticket(s) from the last run - set qty / price / type, "
                       f"then Place. Orders route to your {env} Alpaca account.").pack(
            anchor="w", padx=4, pady=(0, 6))
        for r in recs:
            sym = r["symbol"]
            entry = r.get("entry") or 0
            stop, target = r.get("stop"), r.get("target")
            default_qty = port.get(sym, {}).get("shares") or r.get("shares_at_ref_equity") or 0
            row = ttk.Frame(self.orders_body, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=sym, style="Card.TLabel", width=7).pack(side="left")
            ttk.Label(row, text=f"~${entry}", style="CardMuted.TLabel", width=11).pack(side="left")
            ttk.Label(row, text="qty", style="CardMuted.TLabel").pack(side="left")
            qv = tk.StringVar(value=str(int(default_qty)))
            ttk.Entry(row, textvariable=qv, width=6).pack(side="left", padx=(3, 8))
            tv = tk.StringVar(value="limit")
            ttk.Combobox(row, textvariable=tv, values=["market", "limit"],
                         state="readonly", width=7).pack(side="left", padx=(0, 8))
            pv = tk.StringVar(value=str(entry))
            ttk.Label(row, text="$", style="CardMuted.TLabel").pack(side="left")
            ttk.Entry(row, textvariable=pv, width=8).pack(side="left", padx=(2, 8))
            bv = tk.BooleanVar(value=bool(stop and target))
            ttk.Checkbutton(row, text="stop/target", variable=bv).pack(side="left", padx=(0, 8))
            ttk.Button(row, text="Place", style="Accent.TButton",
                       command=lambda r=r, qv=qv, tv=tv, pv=pv, bv=bv:
                       self._place_order(r, qv, tv, pv, bv)).pack(side="left")

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
                 "target": rec.get("target"), "attach_bracket": bool(bv.get())}
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
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

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
        self._set_busy(True)
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

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.btn_ticker.config(state=state)
        self.btn_screen.config(state=state)
        self.btn_backtest.config(state=state)
        self.btn_recs.config(state=state)
        self.btn_momentum.config(state=state)
        self.btn_reddit.config(state=state)
        self.btn_val.config(state=state)
        self.btn_paper.config(state=state)
        self.btn_delib.config(state=state)
        self.btn_alpaca.config(state=state)
        self.btn_portfolio.config(state=state)
        self.btn_hist.config(state=state)
        self.btn_filings.config(state=state)
        self.spinner.config(text="● running…" if busy else "● ready")

    @staticmethod
    def _tag_for(line: str) -> str | None:
        s = line.strip()
        low = s.lower()
        if s.startswith("!!") or "[error]" in low or "verdict: do not buy" in low:
            return "err"
        if "[warn" in low or "warning" in low or low.startswith("[note]"):
            return "warn"
        if ("verdict: buy" in low or "recommend buy" in low or s.startswith("[done]")
                or "[selftest] ok" in low):
            return "ok"
        if s.startswith("===") or s.startswith("ANALYSIS:") or ">>> AGENT" in s \
                or s.startswith("====") or s.startswith("ANALYSIS") or "ANALYST" in s.upper()[:14]:
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
