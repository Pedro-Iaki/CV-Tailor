# Builds the portable app: dist\CV Tailor\ (and dist\CV-Tailor-portable.zip).
# Usage: powershell -ExecutionPolicy Bypass -File build_portable.ps1
#
# The folder holds CV Tailor.exe and _internal\ (the Python runtime). On first start
# the app creates Data\ (settings.env, then your master.yaml, database, variants) and
# Output\ (PDFs) next to the exe. Nothing personal is ever put in the build.
$ErrorActionPreference = 'Stop'
$py = '.venv\Scripts\python.exe'

& $py -m PyInstaller --noconfirm --clean --windowed --name 'CV Tailor' `
  --add-data 'templates;templates' `
  --add-data 'cvtailor\ui;cvtailor\ui' `
  --hidden-import cvtailor.careful --hidden-import cvtailor.ratings --hidden-import cvtailor.ollama_setup `
  --exclude-module weasyprint --exclude-module tkinter --exclude-module typer `
  --exclude-module mcp --exclude-module cvtailor.mcp_server --exclude-module pytest `
  --exclude-module matplotlib --exclude-module numpy --exclude-module PIL `
  cv_tailor_app.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$app = 'dist\CV Tailor'
@'
CV Tailor (portable)
====================
Run "CV Tailor.exe". Everything stays in this folder:
  Data\     settings.env, master.yaml, applications.db, variants\, worked_examples.yaml
  Output\   the PDFs
Move or copy the whole folder to take it elsewhere. To update, replace
"CV Tailor.exe" and "_internal\" and keep Data\ and Output\.

First start: the Setup tab checks Ollama (and offers to install it), downloads the
recommended model, and can import your existing data folder.
Needs Windows 10 or 11 (WebView2 and Edge, both preinstalled).
'@ | Set-Content -Encoding utf8 "$app\README.txt"

$zip = 'dist\CV-Tailor-portable.zip'
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $app -DestinationPath $zip
$size = (Get-ChildItem $app -Recurse | Measure-Object Length -Sum).Sum / 1MB
"Built: $app ($([math]::Round($size)) MB), zip: $zip ($([math]::Round((Get-Item $zip).Length / 1MB)) MB)"
