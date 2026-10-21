$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& (Join-Path $projectRoot '本地环境\python.exe') -m echo train --index 本地数据/模型视图/index.json --config 02_模型训练/静态ECHO.yaml --output 本地运行/静态ECHO
if ($LASTEXITCODE -ne 0) { throw '训练未完成，检查数据权限与来源参考Q' }
