param([string]$DataRoot = '本地数据/原始/qh')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& (Join-Path $projectRoot '本地环境\python.exe') -m echo audit --root $DataRoot --limit 20
if ($LASTEXITCODE -ne 0) { throw '审计失败' }
