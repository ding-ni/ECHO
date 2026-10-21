$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& (Join-Path $projectRoot '本地环境\python.exe') -m echo train --index 本地数据/模型视图/index.json --config 02_模型训练/静态ECHO.yaml --output 本地运行/无记忆消融
if ($LASTEXITCODE -ne 0) { throw '消融未完成' }
