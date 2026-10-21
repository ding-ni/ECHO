$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& (Join-Path $projectRoot '本地环境\python.exe') -m echo download
if ($LASTEXITCODE -ne 0) { throw '下载尚未完成，检查本地数据中的状态文件后重试' }
