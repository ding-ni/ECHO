$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& (Join-Path $projectRoot '本地环境\python.exe') -m echo demo
if ($LASTEXITCODE -ne 0) { throw '样例运行失败' }
