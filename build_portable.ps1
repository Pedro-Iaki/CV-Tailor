# Builds the portable app: dist\CV Tailor\ (and dist\CV-Tailor-portable.zip).
# Usage: powershell -ExecutionPolicy Bypass -File build_portable.ps1
#
# The folder holds CV Tailor.exe and _internal\ (the Python runtime). On first start
# the app creates Data\ (settings.env, then your master.yaml, database, variants) and
# Output\ (PDFs) next to the exe. Nothing personal is ever put in the build: PyInstaller
# writes to build\stage, only the exe and _internal\ are copied over an existing
# dist\CV Tailor (so a rebuild keeps its Data\ and Output\), and the zip is made from
# the stage, never from dist.
$ErrorActionPreference = 'Stop'
$py = '.venv\Scripts\python.exe'
$stage = 'build\stage'

# PyInstaller logs to stderr; Windows PowerShell turns that into errors under 'Stop'.
$ErrorActionPreference = 'Continue'
& $py -m PyInstaller --noconfirm --clean --windowed --name 'CV Tailor' --distpath $stage `
  --add-data 'templates;templates' `
  --add-data 'cvtailor\ui;cvtailor\ui' `
  --hidden-import cvtailor.careful --hidden-import cvtailor.ratings --hidden-import cvtailor.ollama_setup `
  --exclude-module weasyprint --exclude-module tkinter --exclude-module typer `
  --exclude-module mcp --exclude-module cvtailor.mcp_server --exclude-module pytest `
  --exclude-module matplotlib --exclude-module numpy --exclude-module PIL `
  cv_tailor_app.py 2>&1 | Where-Object { "$_" -match 'ERROR|WARNING: lib not found|completed successfully' } | ForEach-Object { "$_" }
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($code -ne 0) { throw "PyInstaller failed ($code)" }

$built = "$stage\CV Tailor"
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
'@ | Set-Content -Encoding utf8 "$built\README.txt"

# Update dist\CV Tailor in place, leaving Data\ and Output\ alone.
$app = 'dist\CV Tailor'
New-Item -ItemType Directory -Force $app | Out-Null
foreach ($item in 'CV Tailor.exe', '_internal', 'README.txt') {
  if (Test-Path "$app\$item") { Remove-Item -Recurse -Force "$app\$item" }
  Copy-Item -Recurse "$built\$item" "$app\$item"
}

$zip = 'dist\CV-Tailor-portable.zip'
if (Test-Path $zip) { Remove-Item $zip }
& $py -c "import shutil, sys; shutil.make_archive(sys.argv[1][:-4], 'zip', sys.argv[2], 'CV Tailor')" $zip $stage
if ($LASTEXITCODE -ne 0) { throw "zip failed" }
$size = (Get-ChildItem $built -Recurse | Measure-Object Length -Sum).Sum / 1MB
"Built: $app ($([math]::Round($size)) MB, Data\ and Output\ kept), zip: $zip ($([math]::Round((Get-Item $zip).Length / 1MB)) MB)"
