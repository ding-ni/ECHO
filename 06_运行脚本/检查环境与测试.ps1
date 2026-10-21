$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '本地环境\python.exe'
$env:PYTHONUTF8 = '1'
& $projectPython -m echo doctor
if ($LASTEXITCODE -ne 0) { throw '环境诊断失败' }
& $projectPython -m pytest -q --basetemp (Join-Path $projectRoot ('本地运行\测试_' + [guid]::NewGuid().ToString('N'))) -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw '测试失败' }
