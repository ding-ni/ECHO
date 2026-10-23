$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$graphPath = Join-Path $projectRoot '本地数据/模型视图/真实分支河网_v1/index.json'
if (Test-Path -LiteralPath $graphPath) {
    $graph = Get-Content -LiteralPath $graphPath -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Output "真实分支河网：$($graph.stations.Count) 站；失败 $($graph.failures.Count) 站"
}
$processPath = Join-Path $projectRoot '本地运行/完整模型DEM下载_进程_v3.json'
if (Test-Path -LiteralPath $processPath) {
    $record = Get-Content -LiteralPath $processPath -Raw -Encoding utf8 | ConvertFrom-Json
    $running = $null -ne (Get-Process -Id $record.pid -ErrorAction SilentlyContinue)
    Write-Output "DEM下载进程 $($record.pid)：运行中=$running"
}
$demRoot = Join-Path $projectRoot '本地数据/原始/3DEP_本地八站_v1'
if (Test-Path -LiteralPath $demRoot) {
    $assets = Get-Content -LiteralPath (Join-Path $demRoot 'selected_assets.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $completed = @(Get-ChildItem -LiteralPath $demRoot -Filter '*.receipt.json')
    Write-Output "DEM完整核验收据：$($completed.Count)/$($assets.assets.Count)"
    Get-ChildItem -LiteralPath $demRoot -Filter '*.status.json' | ForEach-Object {
        $status = Get-Content -LiteralPath $_.FullName -Raw -Encoding utf8 | ConvertFrom-Json
        [pscustomobject]@{
            文件 = $_.Name
            连续已校验字节 = $status.verified_contiguous_bytes
            总字节 = $status.expected_bytes
            更新时间UTC = $status.updated_utc
        }
    } | Format-Table -AutoSize
}
$accessPath = Join-Path $projectRoot '本地数据/审计/正式空间资料目录_v1/NLDAS_下载访问检查.json'
if (Test-Path -LiteralPath $accessPath) {
    $access = Get-Content -LiteralPath $accessPath -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Output "NLDAS原网格访问：HTTP $($access.status)，需Earthdata登录；未完成下载"
}
