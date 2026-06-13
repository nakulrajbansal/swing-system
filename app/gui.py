"""Tkinter desktop GUI: configure, save, and run the system with live output.

No third-party GUI dependency (Tkinter ships with Python) so the PyInstaller
bundle stays self-contained and cross-platform. The look is a custom flat theme
(built on ttk's "clam") so it is modern and consistent on Windows and macOS.
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from app import APP_NAME, APP_VERSION
from app.config import SECRET_FIELDS, AppConfig
from app.runner import (RunStopped, check_alpaca, clear_stop, place_manual_order,
                        request_stop, run_curation, run_momentum_trade,
                        run_portfolio_status, run_position_review,
                        run_recommendations, run_reddit_scan, run_screen,
                        run_strategy_backtest, run_trade_history, run_watch)

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
    ("n_symbols", "Synthetic universe size (offline demo)", "int"),
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
SIDEBAR = "#0c0f14"     # deepest (left rail; also the titlebar caption color)
BG = "#12161d"          # content background
CARD = "#1a202a"        # card / surface (raised above content)
SURF2 = "#252c39"       # raised controls (buttons, fields)
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
        root.geometry("1120x860")                 # restore-size; opens maximized
        root.minsize(940, 680)
        root.configure(bg=BG)
        try:
            if sys.platform == "win32":
                root.state("zoomed")
            else:
                root.attributes("-zoomed", True)
        except tk.TclError:
            pass
        self._set_app_icon()
        self._setup_style()
        self._build()
        self.root.after(80, self._drain_queue)
        # The DWM caption attributes need the window to exist; apply once now
        # and once after the first map (covers both cold start and restore).
        self._theme_titlebar()
        self.root.after(200, self._theme_titlebar)
        # Keyboard-first flow: the three most-used actions never need the mouse.
        root.bind("<Control-r>", lambda e: None if self.running
                  else self._run_screen_choice())
        root.bind("<Escape>", lambda e: self._stop_run() if self.running else None)
        root.bind("<Control-s>", lambda e: self._save())

    def _theme_titlebar(self):
        """Make the native Windows title bar part of the design: dark caption
        matched to the sidebar, themed border (Windows 10 1809+ / 11 via DWM).
        Without this the OS paints a white bar over a dark app."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            one = ctypes.byref(ctypes.c_int(1))
            for attr in (20, 19):                  # USE_IMMERSIVE_DARK_MODE (old/new id)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, one, 4)

            def cref(hexcol):                      # "#rrggbb" -> COLORREF 0x00BBGGRR
                r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
                return ctypes.byref(ctypes.c_int((b << 16) | (g << 8) | r))

            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, cref(SIDEBAR), 4)  # caption
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 36, cref(INK), 4)      # text
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 34, cref(BORDER), 4)   # border
        except Exception:
            pass                                   # older Windows: dark-mode flag only

    def _set_app_icon(self):
        """Draw the brand mark (accent diamond on dark) as the window icon, so
        the taskbar/titlebar doesn't show the stock Tk feather."""
        try:
            icon = tk.PhotoImage(width=32, height=32)
            icon.put(SIDEBAR, to=(0, 0, 32, 32))
            for y in range(32):
                half = max(0, 13 - abs(y - 16) * 13 // 14)
                if half:
                    icon.put(ACCENT, to=(16 - half, y, 16 + half, y + 1))
            self.root.iconphoto(True, icon)
            self._icon = icon                     # keep a reference alive
        except Exception:
            pass

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
        self.f_mono_bold = tkfont.Font(family=mono, size=10, weight="bold")

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
              background=[("pressed", "#1d2330"), ("active", SURF3),
                          ("disabled", "#171b22")],
              foreground=[("disabled", FAINT)],
              bordercolor=[("active", SURF3)])
        s.configure("Accent.TButton", background=ACCENT, foreground=ACCENT_INK,
                    bordercolor=ACCENT, relief="flat", padding=(18, 9), font=self.f_bold,
                    focuscolor=ACCENT)
        s.map("Accent.TButton",
              background=[("pressed", "#2d8a64"), ("active", ACCENT_DK),
                          ("disabled", "#27483a")],
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
        s.configure("CardFaint.TLabel", background=CARD, foreground=FAINT,
                    font=self.f_sub)
        # Slim themed scrollbar for consoles (the stock Windows bar breaks the
        # dark surface).
        s.configure("Console.Vertical.TScrollbar", background=SURF2,
                    troughcolor=CONSOLE_BG, bordercolor=CONSOLE_BG,
                    arrowcolor=CONSOLE_BG, relief="flat", width=8)
        s.map("Console.Vertical.TScrollbar", background=[("active", SURF3)])
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
        for key, label in (("run", "Desk"), ("performance", "Performance"),
                           ("lessons", "Learning"), ("settings", "Settings")):
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
            ("performance", "Performance", "How the desk's calls have actually done"),
            ("lessons", "Learning", "What the desk has learned from closed trades"),
            ("settings", "Settings", "Credentials, data, strategy parameters")):
            page = ttk.Frame(self._content)
            head = ttk.Frame(page)
            head.pack(fill="x", padx=26, pady=(22, 0))
            left = ttk.Frame(head)
            left.pack(side="left")
            ttk.Label(left, text=title, style="PageTitle.TLabel").pack(anchor="w")
            ttk.Label(left, text=sub, style="PageSub.TLabel").pack(anchor="w", pady=(3, 0))
            if key == "run":                       # context chips, right-aligned
                self._chipbar = ttk.Frame(head)
                self._chipbar.pack(side="right", anchor="n", pady=(4, 0))
            body = ttk.Frame(page)
            body.pack(fill="both", expand=True)
            self._pages[key] = page
            page._body = body

        self._build_run(self._pages["run"]._body)
        self._build_performance(self._pages["performance"]._body)
        self._build_lessons(self._pages["lessons"]._body)
        self._build_config(self._pages["settings"]._body)
        self._refresh_chips()
        self._show("run")

    def _show(self, key: str):
        for k, page in self._pages.items():
            page.pack_forget()
        self._pages[key].pack(fill="both", expand=True)
        for k, item in self._nav_btns.items():
            item.set_active(k == key)
        if key == "lessons":
            self._refresh_lessons()
        elif key == "performance":
            self._refresh_performance()

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

    def _console(self, parent, height: int = 20):
        """Dark text surface with the THEMED slim scrollbar (stock Windows
        scrollbars break the dark design). Returns the Text widget, packed."""
        wrap = ttk.Frame(parent, style="Card.TFrame")
        wrap.pack(fill="both", expand=True)
        txt = tk.Text(wrap, wrap="word", height=height, font=self.f_mono,
                      bg=CONSOLE_BG, fg=CONSOLE_FG, insertbackground=CONSOLE_FG,
                      relief="flat", bd=0, padx=18, pady=14,
                      spacing1=2, spacing3=2)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=txt.yview,
                           style="Console.Vertical.TScrollbar")
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        return txt

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
        self.vars["self_tune_weights"] = tk.BooleanVar(value=self.cfg.self_tune_weights)
        ttk.Checkbutton(opts, text="Self-tune screen weights nightly (trailing "
                                   "walk-forward picks the preset)",
                        variable=self.vars["self_tune_weights"]).pack(anchor="w", padx=8, pady=2)
        self.vars["learn_from_runs"] = tk.BooleanVar(value=self.cfg.learn_from_runs)
        ttk.Checkbutton(opts, text="Learn from runs (reflect on closed trades + recall lessons)",
                        variable=self.vars["learn_from_runs"]).pack(anchor="w", padx=8, pady=2)
        self.vars["auto_approve_lessons"] = tk.BooleanVar(value=self.cfg.auto_approve_lessons)
        ttk.Checkbutton(opts, text="Auto-activate new lessons instantly (skip the AI "
                                   "curator's evidence gate)",
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

        auto = self._card(body, "Automation",
                          "let the desk run itself on weekday afternoons")
        ttk.Label(auto, style="CardMuted.TLabel", wraplength=760, justify="left",
                  text="Two scheduled tasks (weekdays, Windows Task Scheduler): the "
                       "daily run at 4:45 pm reviews open positions then screens your "
                       "saved universe; the watchlist check at 12:30 + 3:30 pm fires "
                       "a notification when a watched name reaches its entry "
                       "trigger.").pack(anchor="w", pady=(0, 8))
        abar = ttk.Frame(auto, style="Card.TFrame")
        abar.pack(fill="x")
        ttk.Button(abar, text="Schedule daily run", style="TButton", cursor="hand2",
                   takefocus=False, command=self._schedule_daily).pack(side="left")
        ttk.Button(abar, text="Remove schedule", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._unschedule_daily).pack(
            side="left", padx=(8, 0))
        self.sched_status = ttk.Label(abar, text="", style="CardMuted.TLabel")
        self.sched_status.pack(side="left", padx=(12, 0))
        self._refresh_sched_status()

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
        rows = self._card(parent, "Trade & monitor",
                          "Review exits manages planned positions (time exits, "
                          "guardian advice, arms missing stops) · Momentum "
                          "auto-trade runs the separate mechanical one-name strategy")
        actions = [
            ("Review exits", lambda: self._start(run_position_review)),
            ("Portfolio P&L", lambda: self._start(run_portfolio_status)),
            ("Momentum auto-trade", lambda: self._start(run_momentum_trade)),
            ("Strategy backtest", lambda: self._start(run_strategy_backtest)),
            ("Reddit sentiment", lambda: self._start(run_reddit_scan)),
        ]
        for i, (label, cmd) in enumerate(actions):
            self._mkbtn(rows, label, cmd).grid(
                row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
        for c in range(len(actions)):
            rows.columnconfigure(c, weight=1, uniform="desk")

        # ---- selective execution: tickets from the last run ----
        self.orders_body = self._card(parent, "Order tickets",
                                      "BUY recommendations from the last run")
        self._orders_hint = ttk.Label(
            self.orders_body, style="CardFaint.TLabel",
            text="Run a screen or a deep-dive — BUY recommendations appear here as "
                 "tickets you can review and place individually.")
        self._orders_hint.pack(anchor="w")

        # ---- watchlist: names waiting for their entry trigger ----
        wb = self._card(parent, "Watchlist", "names waiting for their entry trigger")
        wrow = ttk.Frame(wb, style="Card.TFrame")
        wrow.pack(fill="x")
        self.watch_label = ttk.Label(wrow, style="CardMuted.TLabel", justify="left")
        self.watch_label.pack(side="left", fill="x", expand=True)
        self._mkbtn(wrow, "Check triggers now",
                    lambda: self._start(run_watch)).pack(side="right")
        self._refresh_watchcard()

        # ---- output console with its own toolbar ----
        cons = self._card(parent, "Output",
                          "Ctrl+R runs the selected screen · Esc stops a run",
                          expand=True)
        tools = ttk.Frame(cons, style="Card.TFrame")
        tools.pack(fill="x", pady=(0, 6))
        # Jump-to-section: long analyses become navigable instead of a scroll.
        self._section_idx: dict[str, str] = {}
        self.jumpbox = ttk.Combobox(tools, state="readonly", width=34,
                                    values=["Jump to section…"])
        self.jumpbox.set("Jump to section…")
        self.jumpbox.pack(side="left")
        self.jumpbox.bind("<<ComboboxSelected>>", self._jump_section)
        self.progress = ttk.Progressbar(tools, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar", length=180)
        self.btn_stop = ttk.Button(tools, text="■  Stop run", style="Tool.TButton",
                                   cursor="hand2", takefocus=False,
                                   command=self._stop_run)
        ttk.Button(tools, text="Open logs folder", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._open_logs).pack(side="right", padx=(6, 0))
        self._btn_copy = ttk.Button(tools, text="Copy output", style="Tool.TButton",
                                    cursor="hand2", takefocus=False,
                                    command=self._copy_output)
        self._btn_copy.pack(side="right", padx=(6, 0))
        ttk.Button(tools, text="Clear", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._clear).pack(side="right")
        self.out = self._console(cons)
        self.out.tag_configure("err", foreground=DANGER)
        self.out.tag_configure("warn", foreground="#d8a657")
        self.out.tag_configure("ok", foreground="#57c98a")
        self.out.tag_configure("hl", foreground="#8fb4f0", font=self.f_mono_bold,
                               spacing1=8)
        self.out.tag_configure("agent", foreground=ACCENT, font=self.f_mono_bold,
                               spacing1=6)
        self.out.tag_configure("gem", foreground=GEM)
        self.out.tag_configure("dim", foreground="#677183")
        self.out.tag_configure("rule", foreground="#39414f", spacing1=8, spacing3=4)
        self.out.configure(state="disabled")
        ttk.Frame(parent).pack(pady=5)            # bottom breathing room
        self._log(f"{APP_NAME} ready.  Pick a universe and Run screen (Ctrl+R), or "
                  "deep-dive a single ticker.")
        self._log("    keyboard: Ctrl+R run screen · Esc stop · Ctrl+S save settings"
                  "  ·  credentials live under Settings")

    def _run_screen_choice(self):
        key = self._SCREENS.get(self._screen_choice.get(), "sp500")
        self._start(run_screen, screen_index=key)

    def _refresh_watchcard(self):
        import datetime

        from app import watchlist as wl
        try:
            items = wl.active(datetime.date.today().isoformat())
        except Exception:
            items = []
        if not items:
            text = ("empty - screens add WATCH-tier ideas and the PM's "
                    "wait-for-pullback calls automatically.")
        else:
            parts = []
            for it in items[:8]:
                lvl = (f"dip to {it['pullback_target']}" if it.get("pullback_target")
                       else f"break {it.get('breakout_level')}")
                parts.append(f"{it['symbol']} ({lvl})")
            text = (f"{len(items)} watched:   " + "    ".join(parts)
                    + ("   …" if len(items) > 8 else ""))
        self.watch_label.config(text=text)

    def _copy_output(self):
        text = self.out.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._btn_copy.config(text="✓ Copied")
        self.root.after(1800, lambda: self._btn_copy.winfo_exists()
                        and self._btn_copy.config(text="Copy output"))

    def _build_performance(self, parent):
        curve = self._card(parent, "Equity curve",
                           "cumulative return of scored recommendations, in order")
        self.perf_canvas = tk.Canvas(curve, bg=CONSOLE_BG, height=170,
                                     highlightthickness=0, bd=0)
        self.perf_canvas.pack(fill="x")
        card = self._card(parent, "Scorecard",
                          "calibration, lens cohorts, closed calls, executed trades",
                          expand=True)
        pbar = ttk.Frame(card, style="Card.TFrame")
        pbar.pack(fill="x", pady=(0, 6))
        self._mkbtn(pbar, "Refresh trade history",
                    lambda: self._start(run_trade_history)).pack(side="right")
        self.perf_out = self._console(card, height=18)
        self.perf_out.configure(state="disabled")
        ttk.Frame(parent).pack(pady=5)

    def _refresh_performance(self):
        from app import reco_ledger
        from system.reflection.calibration import calibration_table

        rows = reco_ledger.load()
        scored = sorted((r for r in rows if r.get("status") == "evaluated"
                         and isinstance(r.get("return_pct"), (int, float))),
                        key=lambda r: str(r.get("evaluated_on") or ""))
        # ---- equity curve ----
        c = self.perf_canvas
        c.delete("all")
        self.root.update_idletasks()
        w = max(c.winfo_width(), 420)
        h = 170
        eq, v = [1.0], 1.0
        for r in scored:
            v *= 1 + float(r["return_pct"]) / 100.0
            eq.append(v)
        if len(eq) < 3:
            c.create_text(w // 2, h // 2, fill=MUTED, font=self.f_sub,
                          text="The curve starts once recommendations mature and "
                               "are scored - check back after the first exits.")
        else:
            lo, hi = min(eq + [1.0]), max(eq + [1.0])
            pad = (hi - lo) * 0.12 or 0.05
            lo, hi = lo - pad, hi + pad

            def xy(i, val):
                return (16 + i * (w - 60) / (len(eq) - 1),
                        h - 14 - (val - lo) * (h - 28) / (hi - lo))

            y1 = xy(0, 1.0)[1]
            c.create_line(16, y1, w - 44, y1, fill="#39414f", dash=(3, 4))
            pts = [coord for i, val in enumerate(eq) for coord in xy(i, val)]
            c.create_line(*pts, fill=ACCENT, width=2, smooth=False)
            tot = (eq[-1] - 1) * 100
            c.create_text(w - 40, xy(len(eq) - 1, eq[-1])[1], anchor="w",
                          fill=OK if tot >= 0 else DANGER, font=self.f_sub,
                          text=f"{tot:+.1f}%")
        # ---- scorecard text ----
        lines = []
        if scored:
            wins = sum(1 for r in scored if r["return_pct"] > 0)
            avg = sum(r["return_pct"] for r in scored) / len(scored)
            lines.append(f"SCORED CALLS  {len(scored)}   |   hit rate "
                         f"{100 * wins / len(scored):.0f}%   |   avg {avg:+.2f}%")
            exc = [r["excess_pct"] for r in scored
                   if isinstance(r.get("excess_pct"), (int, float))]
            if exc:
                lines.append(f"vs SPY (same windows): avg excess "
                             f"{sum(exc) / len(exc):+.2f}% over {len(exc)} calls")
            lines.append("")
            lines.append("CALIBRATION (trailing window)")
            table = calibration_table(rows)
            for b in table["bands"]:
                rate = (f"{b['win_rate_pct']:.0f}%" if b["win_rate_pct"] is not None
                        else "  -")
                lines.append(f"  conviction {b['lo']:.2f}-{min(b['hi'], 1.0):.2f}: "
                             f"n={b['n']:<3} realized {rate}")
            lines.append("")
            lines.append("LENS COHORTS")
            for key, label in (("hidden_gem", "hidden-gem"), ("core", "core"),
                               ("moat_bullish", "moat-bullish")):
                co = reco_ledger.cohort_stats(rows).get(key, {})
                if co.get("n"):
                    lines.append(f"  {label:<12} n={co['n']:<3} hit "
                                 f"{co['win_rate_pct']:.0f}%  avg "
                                 f"{co['avg_return_pct']:+.2f}%")
            lines.append("")
            lines.append("RECENT CLOSED CALLS")
            for r in scored[-10:][::-1]:
                gem = " ◆" if r.get("hidden_gem") else ""
                lines.append(f"  {r.get('evaluated_on', '?')}  {r['symbol']:<6}{gem} "
                             f"{r['return_pct']:+6.1f}%  ({r.get('outcome', '?')}, "
                             f"conviction {r.get('conviction', 0):.2f})")
        else:
            n_open = sum(1 for r in rows if r.get("status") == "open")
            lines.append(f"No scored calls yet - {n_open} open recommendation(s) "
                         "awaiting their exit windows.")
        # ---- executed trades (broker fills, cached by Refresh) ----
        import json as _json

        from app.runner import TRADES_CACHE
        lines.append("")
        lines.append("EXECUTED TRADES (your account's actual fills)")
        try:
            trades = (_json.loads(TRADES_CACHE.read_text(encoding="utf-8"))
                      if TRADES_CACHE.exists() else [])
        except Exception:
            trades = []
        if trades:
            lines.append("  when              side  qty    symbol   price")
            for t in trades[-40:]:
                lines.append(f"  {t['when']:<17} {t['side']:<5} {t['qty']:>5.0f}  "
                             f"{t['symbol']:<7} {t['price']:>9.2f}")
            if len(trades) > 40:
                lines.append(f"  ... ({len(trades) - 40} older fill(s) - see the "
                             "console after Refresh trade history)")
        else:
            lines.append("  press 'Refresh trade history' to pull your fills "
                         "from the broker.")
        self.perf_out.configure(state="normal")
        self.perf_out.delete("1.0", "end")
        self.perf_out.insert("end", "\n".join(lines) + "\n")
        self.perf_out.configure(state="disabled")

    def _build_lessons(self, parent):
        card = self._card(parent, "Learning memory",
                          "advisory lessons + base rates that inform future runs",
                          expand=True)
        bar = ttk.Frame(card, style="Card.TFrame")
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, style="CardMuted.TLabel",
                  text="Lessons are reviewed by the AI curator: anecdotes activate "
                       "only once realized results back them; patterns need 5+ "
                       "scored calls.").pack(side="left")
        ttk.Button(bar, text="Refresh", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._refresh_lessons).pack(side="right")
        ttk.Button(bar, text="Clear all", style="Tool.TButton", cursor="hand2",
                   takefocus=False, command=self._clear_lessons).pack(
            side="right", padx=(0, 6))
        b = ttk.Button(bar, text="Run AI curator", style="Tool.TButton",
                       cursor="hand2", takefocus=False,
                       command=lambda: self._start(run_curation))
        b.pack(side="right", padx=(0, 6))
        self._action_buttons.append(b)
        self.lessons_out = self._console(card, height=18)
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
        d["self_tune_weights"] = bool(self.vars["self_tune_weights"].get())
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
        for c, head in enumerate(("symbol", "conv", "P(win)", "ref price", "qty",
                                  "type", "limit $", "bracket", "")):
            ttk.Label(grid, text=head, style="CardMuted.TLabel").grid(
                row=0, column=c, sticky="w", padx=(0, 10), pady=(0, 3))
        for i, r in enumerate(recs, start=1):
            sym = r["symbol"]
            entry = r.get("entry") or 0
            stop, target = r.get("stop"), r.get("target")
            # Risk-sized default; for tiny accounts where 1%-risk rounds to zero,
            # fall back to what the account can actually afford.
            default_qty = (port.get(sym, {}).get("shares")
                           or r.get("shares_at_ref_equity")
                           or r.get("affordable_qty") or 0)
            cell = ttk.Frame(grid, style="Card.TFrame")
            cell.grid(row=i, column=0, sticky="w", padx=(0, 10), pady=5)
            ttk.Label(cell, text=sym, style="Card.TLabel").pack(side="left")
            if r.get("hidden_gem"):
                tk.Label(cell, text=" ◆", bg=CARD, fg=GEM,
                         font=self.f_base).pack(side="left")
            if r.get("already_held"):
                tk.Label(cell, text=" ●", bg=CARD, fg="#d8a657",
                         font=self.f_base).pack(side="left")
            ttk.Label(grid, text=f"{r.get('conviction', 0):.2f}", font=self.f_mono,
                      style="CardMuted.TLabel").grid(row=i, column=1, sticky="w",
                                                     padx=(0, 10), pady=5)
            pwin = r.get("p_win")
            ttk.Label(grid, text=f"{pwin * 100:.0f}%" if pwin else "-",
                      font=self.f_mono,
                      style="CardMuted.TLabel").grid(row=i, column=2, sticky="w",
                                                     padx=(0, 10), pady=5)
            ttk.Label(grid, text=f"~${entry}", font=self.f_mono,
                      style="CardMuted.TLabel").grid(
                row=i, column=3, sticky="w", padx=(0, 10), pady=5)
            qv = tk.StringVar(value=str(int(default_qty)))
            ttk.Entry(grid, textvariable=qv, width=6).grid(
                row=i, column=4, sticky="w", padx=(0, 10), pady=5)
            tv = tk.StringVar(value="limit")
            ttk.Combobox(grid, textvariable=tv, values=["market", "limit"],
                         state="readonly", width=7).grid(
                row=i, column=5, sticky="w", padx=(0, 10), pady=5)
            # The PM's pullback entry (an 'adjust' decision) pre-fills the limit:
            # the most actionable number in the deliberation, not the last close.
            pv = tk.StringVar(value=str(r.get("suggested_entry") or entry))
            ttk.Entry(grid, textvariable=pv, width=9).grid(
                row=i, column=6, sticky="w", padx=(0, 10), pady=5)
            bv = tk.BooleanVar(value=bool(stop and target))
            ttk.Checkbutton(grid, text="stop/target", variable=bv).grid(
                row=i, column=7, sticky="w", padx=(0, 14), pady=5)
            btn_place = ttk.Button(grid, text="Place order", style="Accent.TButton",
                                   cursor="hand2", takefocus=False)
            btn_place.configure(
                command=lambda r=r, qv=qv, tv=tv, pv=pv, bv=bv, b=btn_place:
                self._place_order(r, qv, tv, pv, bv, b))
            btn_place.grid(row=i, column=8, sticky="w", pady=5)
        notes = []
        if any(r.get("hidden_gem") for r in recs):
            notes.append("◆ = hidden-gem pick (early acceleration)")
        if any(r.get("already_held") for r in recs):
            notes.append("● = already an open position (placing adds to it)")
        if any(r.get("suggested_entry") for r in recs):
            notes.append("limit pre-set to the PM's pullback entry where it advised "
                         "waiting for a dip (below the ref price)")
        if notes:
            ttk.Label(self.orders_body, text="   ·   ".join(notes),
                      style="CardMuted.TLabel").pack(anchor="w", padx=2, pady=(6, 0))

    def _place_order(self, rec, qv, tv, pv, bv, btn=None):
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
        if btn is not None:                       # instant feedback on THIS row
            btn.config(text="Placing…", state="disabled")
        self._log(f"\n[order] submitting BUY {qty} {rec['symbol']} ({px}) ...")

        def work():
            res = place_manual_order(cfg, order, lambda line: self.q.put(("log", line)))
            self.q.put(("ticket", (btn, bool(res.get("ok")))))

        threading.Thread(target=work, daemon=True).start()

    def _save(self):
        try:
            cfg = self._collect()
            path = cfg.save()
            self.cfg = cfg
            self.status.config(text=f"✓  Saved  ·  {path}")
            self.root.after(4000, lambda: self.status.winfo_exists()
                            and self.status.config(text=""))
            self._refresh_env_badge()
            self._refresh_chips()
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _refresh_chips(self):
        """Desk-header context chips: what the next run will actually use."""
        for w in self._chipbar.winfo_children():
            w.destroy()

        def chip(text, on):
            tk.Label(self._chipbar, text=text, bg=SURF2 if not on else ACCENT_SOFT,
                     fg=OK if on else MUTED, font=self.f_sub, padx=10, pady=3
                     ).pack(side="left", padx=(6, 0))

        try:
            live = str(self.vars.get("data_source", None) and
                       self.vars["data_source"].get() or self.cfg.data_source) == "live"
            llm = bool(self.vars["use_llm_agents"].get()
                       if "use_llm_agents" in self.vars else self.cfg.use_llm_agents)
            keys = bool((self.vars["alpaca_key_id"].get()
                         if "alpaca_key_id" in self.vars else self.cfg.alpaca_key_id))
        except Exception:
            live, llm, keys = self.cfg.data_source == "live", self.cfg.use_llm_agents, False
        chip("DATA · LIVE" if live else "DATA · SYNTHETIC", live)
        chip("AGENTS · LLM" if llm else "AGENTS · DETERMINISTIC", llm)
        chip("SIZING · REAL ACCOUNT" if keys else "SIZING · CONFIGURED", keys)

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

    # -- daily automation (Windows Task Scheduler) ---------------------------
    _TASK_NAME = "SwingSystem Daily Run"

    def _daily_command(self) -> str | None:
        """The command the scheduled task runs: the packaged exe with --daily."""
        if getattr(sys, "frozen", False):
            return f'"{sys.executable}" --daily'
        from pathlib import Path
        exe = Path(__file__).resolve().parents[1] / "dist" / "SwingSystem.exe"
        return f'"{exe}" --daily' if exe.exists() else None

    _WATCH_TASK = "SwingSystem Watchlist"

    def _schedule_daily(self):
        import subprocess
        if sys.platform != "win32":
            messagebox.showinfo("Windows only", "Task scheduling uses Windows "
                                "Task Scheduler; on macOS use launchd/cron.")
            return
        cmd = self._daily_command()
        if not cmd:
            messagebox.showinfo("Build the app first",
                                "No SwingSystem.exe found to schedule - build it "
                                "with build\\build_windows.ps1, or run the "
                                "packaged app and schedule from there.")
            return
        try:
            subprocess.run(["schtasks", "/Create", "/F", "/SC", "WEEKLY",
                            "/D", "MON,TUE,WED,THU,FRI", "/TN", self._TASK_NAME,
                            "/TR", cmd, "/ST", "16:45"],
                           check=True, capture_output=True, text=True)
            # Watchlist triggers fire DURING market hours: 12:30 + every 3h for
            # 4h (=> 12:30 and 15:30) on weekdays.
            subprocess.run(["schtasks", "/Create", "/F", "/SC", "WEEKLY",
                            "/D", "MON,TUE,WED,THU,FRI", "/TN", self._WATCH_TASK,
                            "/TR", cmd.replace("--daily", "--watch"),
                            "/ST", "12:30", "/RI", "180", "/DU", "04:00"],
                           check=True, capture_output=True, text=True)
        except Exception as exc:
            err = getattr(exc, "stderr", "") or str(exc)
            messagebox.showerror("Scheduling failed", err.strip()[:400])
            return
        self._refresh_sched_status()

    def _unschedule_daily(self):
        import subprocess
        for name in (self._TASK_NAME, self._WATCH_TASK):
            try:
                subprocess.run(["schtasks", "/Delete", "/F", "/TN", name],
                               check=True, capture_output=True, text=True)
            except Exception:
                pass
        self._refresh_sched_status()

    def _refresh_sched_status(self):
        import subprocess
        try:
            r = subprocess.run(["schtasks", "/Query", "/TN", self._TASK_NAME],
                               capture_output=True, text=True)
            on = r.returncode == 0
        except Exception:
            on = False
        self.sched_status.config(
            text="● scheduled - weekdays 4:45 pm" if on else "not scheduled",
            foreground=OK if on else MUTED)

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
        clear_stop()                               # fresh run, fresh stop flag
        task = fn.__name__.replace("run_", "").replace("_", " ")
        self._show("run")                          # output lives on the Desk
        self._set_busy(True, task)
        self._log(f"\n=== starting: {fn.__name__} ===")

        def work():
            try:
                res = fn(cfg, lambda line: self.q.put(("log", line)))
                self.q.put(("result", res))
                self.q.put(("done", None))
            except RunStopped:
                self.q.put(("log", "[stopped] run cancelled by user - no further "
                                   "actions taken; partial output above."))
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
                elif kind == "ticket":             # per-row order feedback
                    btn, ok = payload
                    if btn is not None and btn.winfo_exists():
                        btn.config(text="✓ Sent" if ok else "✗ Failed")
                        self.root.after(3000, lambda b=btn: b.winfo_exists() and
                                        b.config(text="Place order", state="normal"))
                elif kind == "done":
                    self.running = False
                    self._set_busy(False)
                    self._refresh_lessons()    # curator may have changed the set
                    self._refresh_watchcard()  # screens/watch runs change it
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
            self.btn_stop.config(state="normal", text="■  Stop run")
            self.btn_stop.pack(side="left", padx=(10, 0))
            self._tick_spinner()
        else:
            self._busy_task = ""
            self.progress.stop()
            self.progress.pack_forget()
            self.btn_stop.pack_forget()
            self.spinner.config(text="●  ready")

    def _stop_run(self):
        """Cooperative cancel: the run aborts at its next step (log line)."""
        request_stop()
        self.btn_stop.config(state="disabled", text="■  stopping…")
        self._log("[stop] stop requested - the run will halt at its next step "
                  "(a long download chunk may take a moment to finish).")

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
                or "[risk]" in low or "verdict: do not buy" in low):
            return "err"
        if "[warn" in low or "warning" in low or low.startswith("[note]") \
                or "[hint]" in low or "[stop]" in low or "[stopped]" in low:
            return "warn"
        if "hidden gem" in low or "hidden-gem" in low:
            return "gem"
        if ("verdict: buy" in low or "recommend buy" in low or s.startswith("[done]")
                or "[selftest] ok" in low):
            return "ok"
        if s.startswith("▸"):
            return "agent"
        if (s.startswith("===") or s.startswith("ANALYSIS")
                or s.isupper() and len(s) > 3 and not s.startswith("[")):
            return "hl"
        if s.startswith("[") or s.startswith("  -") or s.startswith("  +") \
                or s.startswith("- ") or s.startswith("+ "):
            return "dim"
        return None

    _SECTION_PAT = re.compile(
        r"^(DEEP-DIVE: \S+.*|TRADE RECOMMENDATIONS.*|.*SCREEN RESULTS.*|"
        r"starting: \S+|TOP IDEAS.*|SUGGESTED PORTFOLIO.*|"
        r"[A-Z][A-Z0-9.\-]{0,6}\s{3}(?:BUY|DO NOT BUY).*)$")

    def _register_section(self, disp: str, index: str) -> None:
        m = self._SECTION_PAT.match(disp.strip())
        if not m:
            return
        label = disp.strip()
        if label.startswith("starting:"):
            self._section_idx.clear()              # new run: fresh outline
            label = "▶ " + label
        elif "BUY" in label and not label.startswith(("TRADE", "TOP")):
            label = "verdict — " + label.split()[0]
        label = label[:44]
        n = 2
        base = label
        while label in self._section_idx:          # keep duplicates addressable
            label = f"{base} ({n})"
            n += 1
        self._section_idx[label] = index
        self.jumpbox.configure(values=["Jump to section…"]
                               + list(self._section_idx))

    def _jump_section(self, _event=None):
        idx = self._section_idx.get(self.jumpbox.get())
        if idx:
            self.out.see(idx)
            self.out.yview_moveto(
                float(self.out.index(idx).split(".")[0])
                / max(float(self.out.index("end-1c").split(".")[0]), 1.0))
        self.jumpbox.set("Jump to section…")

    def _log(self, line: str):
        """Render a backend log line for humans: banner rows become thin rules,
        section markers become headers; everything else is tag-colored. Section
        lines register in the jump-to outline."""
        s = line.rstrip()
        stripped = s.strip()
        if stripped and set(stripped) <= {"#", "=", "-"} and len(stripped) >= 8:
            disp, tag = "─" * 78, "rule"
        elif stripped.startswith("# ") or stripped.startswith("=== "):
            disp, tag = stripped.lstrip("#= ").rstrip(" =#"), "hl"
        else:
            disp, tag = s, self._tag_for(s)
        self.out.configure(state="normal")
        index = self.out.index("end-1c linestart")
        self.out.insert("end", disp + "\n", tag or ())
        self.out.see("end")
        self.out.configure(state="disabled")
        self._register_section(disp, index)

    def _clear(self):
        self._section_idx.clear()
        self.jumpbox.configure(values=["Jump to section…"])
        self.jumpbox.set("Jump to section…")
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
