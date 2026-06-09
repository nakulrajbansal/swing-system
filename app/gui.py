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
from app.runner import (check_alpaca, run_deliberation, run_filing_validation,
                        run_insider_validation, run_momentum_trade, run_paper,
                        run_recommendations, run_reddit_scan, run_screen,
                        run_validation)

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

# -- palette (dark, elegant "terminal": charcoal surfaces, one emerald accent) --
BG = "#14171c"          # app background (charcoal)
CARD = "#1b1f27"        # card / surface
SURF2 = "#222733"       # raised controls (buttons, fields)
INK = "#e6e9ef"         # primary text (off-white)
MUTED = "#8b94a7"       # secondary text
ACCENT = "#3fb27f"      # emerald accent (used sparingly)
ACCENT_DK = "#34966b"
ACCENT_INK = "#0e1216"  # text on the accent
OK = "#57c98a"
BORDER = "#2a3039"
CONSOLE_BG = "#0f1216"  # near-black console
CONSOLE_FG = "#cdd3de"


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
        self.f_title = tkfont.Font(family=fam, size=15, weight="bold")
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
        s.configure("Header.TFrame", background=BG)
        s.configure("Rule.TFrame", background=BORDER)
        s.configure("TLabel", background=BG, foreground=INK)
        s.configure("Card.TLabel", background=CARD, foreground=INK)
        s.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=self.f_sub)
        s.configure("Header.TLabel", background=BG, foreground=INK, font=self.f_title)
        s.configure("HeaderSub.TLabel", background=BG, foreground=MUTED, font=self.f_sub)
        s.configure("Accent.TLabel", background=BG, foreground=ACCENT, font=self.f_sub)
        s.configure("Status.TLabel", background=BG, foreground=MUTED, font=self.f_sub)
        s.configure("OK.TLabel", background=BG, foreground=OK, font=self.f_sub)

        s.configure("TButton", background=SURF2, foreground=INK, bordercolor=BORDER,
                    relief="flat", padding=(13, 8), font=self.f_base)
        s.map("TButton",
              background=[("active", "#2c333f"), ("disabled", "#191d24")],
              foreground=[("disabled", "#5b6573")],
              bordercolor=[("active", ACCENT)])
        s.configure("Accent.TButton", background=ACCENT, foreground=ACCENT_INK,
                    bordercolor=ACCENT, relief="flat", padding=(15, 9), font=self.f_bold)
        s.map("Accent.TButton",
              background=[("active", ACCENT_DK), ("disabled", "#2c4a3d")],
              foreground=[("disabled", "#7f8b85")])

        s.configure("TCheckbutton", background=CARD, foreground=INK, font=self.f_base)
        s.map("TCheckbutton", background=[("active", CARD)],
              foreground=[("disabled", MUTED)],
              indicatorcolor=[("selected", ACCENT), ("!selected", SURF2)])
        s.configure("TEntry", fieldbackground=SURF2, foreground=INK, bordercolor=BORDER,
                    insertcolor=INK, relief="flat", padding=5)
        s.map("TEntry", bordercolor=[("focus", ACCENT)])
        s.configure("TCombobox", fieldbackground=SURF2, foreground=INK, bordercolor=BORDER,
                    arrowcolor=MUTED, padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", SURF2)],
              foreground=[("readonly", INK)], bordercolor=[("focus", ACCENT)])

        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(6, 6, 6, 0))
        s.configure("TNotebook.Tab", background=BG, foreground=MUTED,
                    padding=(18, 9), font=self.f_base, borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", BG)],
              foreground=[("selected", ACCENT), ("active", INK)])

        s.configure("TLabelframe", background=CARD, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
        s.configure("TLabelframe.Label", background=CARD, foreground=MUTED,
                    font=self.f_section)
        s.configure("Vertical.TScrollbar", background=SURF2, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED, relief="flat")
        s.map("Vertical.TScrollbar", background=[("active", "#333b48")])

    # -- layout ------------------------------------------------------------
    def _build(self):
        header = ttk.Frame(self.root, style="Header.TFrame")
        header.pack(fill="x")
        inner = ttk.Frame(header, style="Header.TFrame")
        inner.pack(fill="x", padx=20, pady=(14, 10))
        ttk.Label(inner, text="◎  " + APP_NAME, style="Header.TLabel").pack(side="left")
        ttk.Label(inner, text=f"   v{APP_VERSION}", style="HeaderSub.TLabel").pack(
            side="left", pady=(6, 0))
        ttk.Label(inner, text="AI swing desk", style="Accent.TLabel").pack(
            side="right", pady=(6, 0))
        ttk.Frame(header, style="Rule.TFrame", height=1).pack(fill="x")

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        cfg_tab = ttk.Frame(nb)
        run_tab = ttk.Frame(nb)
        lessons_tab = ttk.Frame(nb)
        nb.add(run_tab, text="  Run  ")
        nb.add(lessons_tab, text="  Lessons  ")
        nb.add(cfg_tab, text="  Configuration  ")
        nb.select(run_tab)

        self._build_config(cfg_tab)
        self._build_run(run_tab)
        self._build_lessons(lessons_tab)
        nb.bind("<<NotebookTabChanged>>",
                lambda e: self._refresh_lessons() if nb.tab(nb.select(), "text").strip()
                == "Lessons" else None)

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
        self.btn_paper = ttk.Button(bar2, text="Paper backtest",
                                    command=lambda: self._start(run_paper))
        self.btn_paper.pack(side="left", padx=(0, 8))
        self.btn_hist = ttk.Button(bar2, text="Validate history: insider",
                                   command=lambda: self._start(run_insider_validation))
        self.btn_hist.pack(side="left", padx=(0, 8))
        self.btn_filings = ttk.Button(bar2, text="Validate history: filings",
                                      command=lambda: self._start(run_filing_validation))
        self.btn_filings.pack(side="left", padx=(0, 8))
        self.btn_alpaca = ttk.Button(bar2, text="Check Alpaca",
                                     command=lambda: self._start(check_alpaca))
        self.btn_alpaca.pack(side="left", padx=(0, 8))
        ttk.Button(bar2, text="Clear", command=self._clear).pack(side="left", padx=(0, 8))
        ttk.Button(bar2, text="Open logs", command=self._open_logs).pack(side="left")

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

        # Status / spinner bar.
        sbar = ttk.Frame(parent)
        sbar.pack(fill="x", padx=14, pady=(0, 10))
        self.spinner = ttk.Label(sbar, text="ready", style="Status.TLabel")
        self.spinner.pack(side="left")
        self._log(f"{APP_NAME} ready. Set a ticker and Analyze, or Find trade "
                  "recommendations. Configure keys on the Configuration tab.")

    def _build_lessons(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=14, pady=(12, 4))
        ttk.Label(bar, text="What the desk has learned from closed trades "
                  "(advisory; informs the agents on future runs).",
                  style="Status.TLabel").pack(side="left")
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
            text = summarize(load_memory())
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
                fn(cfg, lambda line: self.q.put(("log", line)))
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
        self.btn_recs.config(state=state)
        self.btn_momentum.config(state=state)
        self.btn_reddit.config(state=state)
        self.btn_val.config(state=state)
        self.btn_paper.config(state=state)
        self.btn_delib.config(state=state)
        self.btn_alpaca.config(state=state)
        self.btn_hist.config(state=state)
        self.btn_filings.config(state=state)
        self.spinner.config(text="running…" if busy else "ready")

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
