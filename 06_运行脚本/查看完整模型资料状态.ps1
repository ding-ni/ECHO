$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$graphPath = Join-Path $projectRoot '本地数据/模型视图/真实分支河网_v1/index.json'
if (Test-Path -LiteralPath $graphPath) {
    $graph = Get-Content -LiteralPath $graphPath -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Output "真实分支河网：$($graph.stations.Count) 站；失败 $($graph.failures.Count) 站"
}
$processPath = Join-Path $projectRoot '本地运行/完整模型DEM下载_进程_v4.json'
if (-not (Test-Path -LiteralPath $processPath)) {
    $processPath = Join-Path $projectRoot '本地运行/完整模型DEM下载_进程_v3.json'
}
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
$accessPath = Join-Path $projectRoot '本地数据/审计/NLDAS_认证小样例/NLDAS_认证验证.json'
if (Test-Path -LiteralPath $accessPath) {
    $access = Get-Content -LiteralPath $accessPath -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Output "NLDAS原网格访问：HTTP $($access.http_status)，状态=$($access.status)"
}
$nasaProgress = Join-Path $projectRoot '本地数据/原始/NLDAS_FORA0125_H_v2/current_progress.json'
if (Test-Path -LiteralPath $nasaProgress) {
    $progress = Get-Content -LiteralPath $nasaProgress -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Output "NLDAS当前批次：已核验 $($progress.completed)/$($progress.plan.hours) 小时；失败 $($progress.failed)"
}
$nodeWeather = Join-Path $projectRoot '本地数据/模型视图/真实原网格气象_首日_v1/index.json'
if (Test-Path -LiteralPath $nodeWeather) {
    $weather = Get-Content -LiteralPath $nodeWeather -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Output "真实增量节点气象：$($weather.stations.Count) 站；首站 $($weather.stations[0].hours) 小时（尚不足完整训练时段）"
}
$envPath = Join-Path $projectRoot '本地数据/模型视图/独立静态与历史气候_v1/index.json'
if (Test-Path -LiteralPath $envPath) {
    $environment = Get-Content -LiteralPath $envPath -Raw -Encoding utf8 | ConvertFrom-Json
    Write-Output "独立静态资料及64维历史气候：$($environment.stations.Count) 站；档案冰川正覆盖 $($environment.positive_glacier_attribute_stations) 站"
}
