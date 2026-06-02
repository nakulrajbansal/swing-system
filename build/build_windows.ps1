# Build the Windows executable (run from the repo root in PowerShell):
#   .\build\build_windows.ps1
# Produces dist\SwingSystem.exe  (single-file, windowed).
# Set $env:CONSOLE = "1" first for a console build that prints logs (debugging).

$ErrorActionPreference = "Stop"

# Use the project venv if present, else the current python.
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

Write-Host "Installing build + app dependencies (incl. real data + LLM)..."
& $py -m pip install -e ".[dev,gui,live-data,llm]" | Out-Null

Write-Host "Building executable with PyInstaller..."
& $py -m PyInstaller build\swing_app.spec --noconfirm --clean --workpath .pyinstaller --distpath dist

Write-Host ""
Write-Host "Done. Executable at: dist\SwingSystem.exe"
Write-Host "Verify (headless): .\dist\SwingSystem.exe --selftest ; type `$env:TEMP\swing_selftest.log"
