$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent $PSScriptRoot
Get-Content -LiteralPath (Join-Path $projectRoot '本地运行\阶段状态.json') -Encoding utf8
Get-ChildItem -LiteralPath (Join-Path $projectRoot '本地数据\原始') -Recurse -Filter '*.status.json' -File | ForEach-Object {
    Write-Output $_.FullName
    Get-Content -LiteralPath $_.FullName -Encoding utf8
}
Get-Content -LiteralPath (Join-Path $projectRoot '本地运行\环境与测试\执行状态.json') -Encoding utf8
