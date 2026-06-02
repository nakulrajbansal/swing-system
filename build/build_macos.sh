#!/usr/bin/env bash
# Build the macOS app bundle (run from the repo root on a Mac):
#   bash build/build_macos.sh
# Produces dist/SwingSystem.app  (double-clickable) and dist/SwingSystem (binary).
# PyInstaller cannot cross-compile: this must run ON macOS.
set -euo pipefail

PY="python3"
if [ -x "./.venv/bin/python" ]; then PY="./.venv/bin/python"; fi

echo "Installing build + app dependencies..."
"$PY" -m pip install -e ".[dev,gui]" >/dev/null

echo "Building app bundle with PyInstaller..."
"$PY" -m PyInstaller build/swing_app.spec --noconfirm --clean --workpath .pyinstaller --distpath dist

echo ""
echo "Done. App at: dist/SwingSystem.app"
echo "Verify (headless): ./dist/SwingSystem --selftest ; cat \"\$TMPDIR/swing_selftest.log\""
echo "Note: unsigned apps may need: right-click -> Open (Gatekeeper), or"
echo "  xattr -dr com.apple.quarantine dist/SwingSystem.app"
