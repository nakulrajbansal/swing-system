# Desktop app — build & run

A Tkinter GUI (`app/`) to configure API keys + run parameters and run the
validation harness / paper-trading engine with live output. Packaged into a
standalone executable with PyInstaller.

## Run from source (no packaging)

```bash
pip install -e ".[dev,gui]"
python -m app.main                 # launch the GUI
python -m app.main --selftest      # headless smoke test (no display needed)
```

## Build a standalone executable

PyInstaller does **not** cross-compile — build on the OS you want to target.

**Windows** (PowerShell, from the repo root):
```powershell
.\build\build_windows.ps1
# -> dist\SwingSystem.exe   (single-file, windowed)
```

**macOS** (Terminal, from the repo root):
```bash
bash build/build_macos.sh
# -> dist/SwingSystem.app   (double-clickable) and dist/SwingSystem (binary)
```

Both wrap `pyinstaller build/swing_app.spec`. For a debuggable build that opens a
console with logs, set `CONSOLE=1` first (`$env:CONSOLE="1"` on Windows,
`export CONSOLE=1` on macOS).

## Verify a build (headless)

```powershell
.\dist\SwingSystem.exe --selftest      # exit code 0 == bundle OK
type $env:TEMP\swing_selftest.log      # results (macOS: $TMPDIR/swing_selftest.log)
```

## Using the app

1. **Configuration tab** — enter any API keys (saved locally to
   `~/.swing_system/config.json`, never bundled or committed) and tune the run
   parameters (universe size, dates, starting equity). Click **Save**.
2. **Run tab** — **Run validation harness** scores the edges (PASS/KILL);
   **Run paper trading** runs the end-to-end engine. Output streams live.

### What's active vs. reserved
- **Active now:** the deterministic, offline pipeline on a synthetic universe.
- **Reserved / gated:** live data source, real-LLM agents (experimental toggle),
  and live-broker trading (asymmetric-autonomy invariant — a human must wire it).
  These are configurable but intentionally not driven by one-click runs.

## Notes
- First launch of the single-file build is slower (it unpacks to a temp dir).
- The bundle is large (~200–400 MB) because it ships pandas/numpy/scipy/pyarrow.
- macOS Gatekeeper may block an unsigned app: right-click → Open, or
  `xattr -dr com.apple.quarantine dist/SwingSystem.app`.
