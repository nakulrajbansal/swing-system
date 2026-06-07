"""Tkinter desktop GUI: configure, save, and run the system with live output.

No third-party GUI dependency (Tkinter ships with Python) so the PyInstaller
bundle stays self-contained and cross-platform.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from app import APP_NAME, APP_VERSION
from app.config import SECRET_FIELDS, AppConfig
from app.runner import (check_alpaca, run_deliberation, run_insider_validation,
                        run_paper, run_validation)

# (field, label, kind)  kind: "secret" | "text" | "int" | "float" | "choice"
_FIELDS = [
    ("anthropic_api_key", "Anthropic API key (LLM agents)", "secret"),
    ("alpaca_key_id", "Alpaca key id (broker)", "secret"),
    ("alpaca_secret", "Alpaca secret (broker)", "secret"),
    ("alpaca_env", "Alpaca environment", "choice"),
    ("edgar_user_agent", "EDGAR User-Agent (e.g. you@email.com)", "text"),
    ("data_source", "Data source", "choice"),
    ("n_symbols", "Universe size (symbols)", "int"),
    ("start_date", "Start date (YYYY-MM-DD)", "text"),
    ("end_date", "End date (YYYY-MM-DD)", "text"),
    ("seed", "Random seed", "int"),
    ("starting_equity", "Starting equity ($)", "float"),
    ("oos_start", "Out-of-sample start (YYYY-MM-DD)", "text"),
    ("insider_history_quarters", "Insider history (quarters)", "int"),
]

# Allowed values for "choice" fields.
_CHOICES = {"data_source": ["synthetic", "live"], "alpaca_env": ["paper", "live"]}


class SwingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = AppConfig.load()
        self.vars: dict[str, tk.Variable] = {}
        self.q: queue.Queue = queue.Queue()
        self.running = False

        root.title(f"{APP_NAME} {APP_VERSION}")
        root.geometry("900x680")
        root.minsize(760, 560)
        self._build()
        self.root.after(80, self._drain_queue)

    # -- layout ------------------------------------------------------------
    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        cfg_tab = ttk.Frame(nb)
        run_tab = ttk.Frame(nb)
        nb.add(cfg_tab, text="Configuration")
        nb.add(run_tab, text="Run")

        self._build_config(cfg_tab)
        self._build_run(run_tab)

    def _build_config(self, parent):
        form = ttk.Frame(parent)
        form.pack(fill="x", padx=12, pady=12)
        for i, (field, label, kind) in enumerate(_FIELDS):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=4, padx=(0, 10))
            cur = getattr(self.cfg, field)
            if kind == "choice":
                var = tk.StringVar(value=str(cur))
                ttk.Combobox(form, textvariable=var, values=_CHOICES.get(field, []),
                             state="readonly", width=38).grid(row=i, column=1, sticky="w")
            else:
                var = tk.StringVar(value=str(cur))
                show = "*" if kind == "secret" else ""
                ttk.Entry(form, textvariable=var, width=42, show=show).grid(
                    row=i, column=1, sticky="w")
            self.vars[field] = var
        form.columnconfigure(1, weight=1)

        toggles = ttk.Frame(parent)
        toggles.pack(fill="x", padx=12, pady=(0, 8))
        self.vars["use_llm_agents"] = tk.BooleanVar(value=self.cfg.use_llm_agents)
        ttk.Checkbutton(toggles, text="Use LLM agents (experimental — may spend tokens)",
                        variable=self.vars["use_llm_agents"]).pack(anchor="w")
        self.vars["only_validated_edges"] = tk.BooleanVar(value=self.cfg.only_validated_edges)
        ttk.Checkbutton(toggles, text="Only trade validated edges (run validation first)",
                        variable=self.vars["only_validated_edges"]).pack(anchor="w")
        self.vars["place_orders"] = tk.BooleanVar(value=self.cfg.place_orders)
        ttk.Checkbutton(toggles, text="Place approved orders on Alpaca (live deliberation)",
                        variable=self.vars["place_orders"]).pack(anchor="w")
        self.vars["enable_live_trading"] = tk.BooleanVar(value=self.cfg.enable_live_trading)
        ttk.Checkbutton(toggles, text="Enable LIVE (real-money) Alpaca env — extra gate",
                        variable=self.vars["enable_live_trading"],
                        command=self._warn_live).pack(anchor="w")

        ttk.Label(parent, foreground="#888", wraplength=820, justify="left",
                  text="Data source 'live' pulls REAL free data (Yahoo), cached locally; "
                       "'synthetic' is a planted-signal demo. Alpaca environment 'paper' = "
                       "fake money, 'live' = REAL money (also requires the Enable-live gate). "
                       "'Place approved orders on Alpaca' submits the live-deliberation's "
                       "approved trades as bracket orders; OFF = show proposals only. LLM "
                       "agents call Anthropic when enabled (capped). Keys are saved to "
                       "~/.swing_system/config.json (never committed or bundled).").pack(
            anchor="w", padx=12, pady=(4, 8))

        btns = ttk.Frame(parent)
        btns.pack(fill="x", padx=12, pady=8)
        ttk.Button(btns, text="Save configuration", command=self._save).pack(side="left")
        self.status = ttk.Label(btns, text="", foreground="#2a7")
        self.status.pack(side="left", padx=12)

    def _build_run(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=12, pady=12)
        self.btn_val = ttk.Button(bar, text="Run validation harness",
                                  command=lambda: self._start(run_validation))
        self.btn_val.pack(side="left")
        self.btn_paper = ttk.Button(bar, text="Run paper trading",
                                    command=lambda: self._start(run_paper))
        self.btn_paper.pack(side="left", padx=8)
        self.btn_delib = ttk.Button(bar, text="Run live deliberation (1 day, LLM)",
                                    command=lambda: self._start(run_deliberation))
        self.btn_delib.pack(side="left", padx=(0, 8))
        self.btn_alpaca = ttk.Button(bar, text="Check Alpaca connection",
                                     command=lambda: self._start(check_alpaca))
        self.btn_alpaca.pack(side="left", padx=(0, 8))
        self.btn_hist = ttk.Button(bar, text="Validate on history (insider)",
                                   command=lambda: self._start(run_insider_validation))
        self.btn_hist.pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Clear output", command=self._clear).pack(side="left")
        ttk.Button(bar, text="Open logs folder", command=self._open_logs).pack(side="left", padx=8)
        self.spinner = ttk.Label(bar, text="", foreground="#27a")
        self.spinner.pack(side="left", padx=12)

        self.out = scrolledtext.ScrolledText(parent, wrap="word", height=26,
                                             font=("Consolas", 9))
        self.out.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.out.configure(state="disabled")
        self._log(f"{APP_NAME} ready. Configure on the first tab, then run here.")

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
        d["place_orders"] = bool(self.vars["place_orders"].get())
        d["enable_live_trading"] = bool(self.vars["enable_live_trading"].get())
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

    def _start(self, fn):
        if self.running:
            return
        try:
            cfg = self._collect()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
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
        self.btn_val.config(state=state)
        self.btn_paper.config(state=state)
        self.btn_delib.config(state=state)
        self.btn_alpaca.config(state=state)
        self.btn_hist.config(state=state)
        self.spinner.config(text="running…" if busy else "")

    def _log(self, line: str):
        self.out.configure(state="normal")
        self.out.insert("end", line + "\n")
        self.out.see("end")
        self.out.configure(state="disabled")

    def _clear(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")

    def _open_logs(self):
        import os
        import subprocess
        import sys
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
