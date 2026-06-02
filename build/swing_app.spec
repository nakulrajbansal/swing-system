# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Swing System desktop app.

Builds a single-file, windowed executable that bundles the GUI and the full
harness/system packages plus their scientific dependencies. Run from the repo
root:  pyinstaller build/swing_app.spec --noconfirm

Cross-platform: run this same spec on Windows to get SwingSystem.exe, or on
macOS to get SwingSystem.app (PyInstaller does not cross-compile; build on the
target OS). Set CONSOLE=1 in the environment for a debuggable console build.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Paths in a .spec resolve relative to the spec's directory; anchor to repo root
# (the parent of build/) so this works regardless of the invoking cwd.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

datas, binaries, hiddenimports = [], [], []

# scipy and pyarrow ship data/extension modules PyInstaller can miss; collect all.
for pkg in ("scipy", "pyarrow"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Our own packages (entry imports them lazily / by string in places).
for pkg in ("harness", "system", "app"):
    hiddenimports += collect_submodules(pkg)

# pandas/numpy are handled by PyInstaller's bundled hooks.
hiddenimports += ["pandas", "numpy"]

console = os.environ.get("CONSOLE", "0") == "1"
is_mac = sys.platform == "darwin"

a = Analysis(
    [os.path.join(ROOT, "app", "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "PyQt5", "PySide6", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SwingSystem",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=console,
    disable_windowed_traceback=False,
)

if is_mac:
    app = BUNDLE(
        exe,
        name="SwingSystem.app",
        icon=None,
        bundle_identifier="com.swingsystem.app",
    )
