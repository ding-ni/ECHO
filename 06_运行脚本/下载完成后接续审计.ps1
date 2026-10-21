$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& (Join-Path $projectRoot '本地环境\python.exe') -m echo.workflow --wait
if ($LASTEXITCODE -ne 0) { throw '接续审计失败，检查日志；原始文件保留' }
