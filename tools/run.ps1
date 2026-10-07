# Run inside this checkout with the existing or newly created Python environment.
# Usage: powershell -File tools/run.ps1 -m pytest -q
$ErrorActionPreference = 'Stop'
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if ($env:OS -eq 'Windows_NT' -and -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio C++ Build Tools.' }
    $vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    $envLines = & cmd.exe /d /c "call `"$vcvars`" >nul && set"
    foreach ($entry in $envLines) {
        if ($entry -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}
$env:DISTUTILS_USE_SDK = '1'
$env:MAX_JOBS = '4'
$env:PYTHONPATH = Join-Path $PSScriptRoot '..'
$pythonExe = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = (Get-Command python.exe).Source }
& $pythonExe @args
exit $LASTEXITCODE
