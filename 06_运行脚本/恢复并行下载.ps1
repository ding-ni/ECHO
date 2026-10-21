$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '本地环境\python.exe'
$env:PYTHONUTF8 = '1'
& $projectPython -m echo download --assets info attributes shapefiles
if ($LASTEXITCODE -ne 0) { throw '小文件校验或下载失败' }
& $projectPython -m echo download --assets qh --connections 4 --output 本地数据/原始/QH独立下载
if ($LASTEXITCODE -ne 0) { throw 'Q/H未完成，可重新运行恢复' }
& $projectPython -m echo download --assets weather --connections 4
if ($LASTEXITCODE -ne 0) { throw '气象未完成，可重新运行恢复' }
& $projectPython -m echo.workflow
