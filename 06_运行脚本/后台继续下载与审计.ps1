$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '本地环境\python.exe'
$stateFile = Join-Path $projectRoot '本地运行\后台任务.json'
$env:PYTHONUTF8 = '1'
$tasks = @()
if (Test-Path -LiteralPath $stateFile) {
    $previous = Get-Content -LiteralPath $stateFile -Encoding utf8 | ConvertFrom-Json
    foreach ($task in $previous) {
        $process = Get-Process -Id $task.pid -ErrorAction SilentlyContinue
        if ($process -and $process.StartTime.ToUniversalTime().ToString('o') -eq $task.startedUtc) {
            $tasks += $task
        }
    }
}
$definitions = @(
    @{ name='weather_download'; args=@('-m','echo.data.download','--assets','weather','--connections','4'); log='气象后台下载' },
    @{ name='post_download_audit'; args=@('-m','echo.workflow','--wait'); log='后台接续审计' }
)
foreach ($definition in $definitions) {
    if ($tasks.name -contains $definition.name) { continue }
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $outFile = Join-Path $projectRoot ('本地缓存\' + $definition.log + '_' + $stamp + '.log')
    $errFile = Join-Path $projectRoot ('本地缓存\' + $definition.log + '_' + $stamp + '.err.log')
    $process = Start-Process -FilePath $projectPython -ArgumentList $definition.args -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $outFile -RedirectStandardError $errFile -PassThru
    $tasks += [PSCustomObject]@{ name=$definition.name; pid=$process.Id; startedUtc=$process.StartTime.ToUniversalTime().ToString('o'); stdout=$outFile; stderr=$errFile }
}
ConvertTo-Json -InputObject @($tasks) -Depth 4 | Set-Content -LiteralPath $stateFile -Encoding utf8
$tasks | Format-Table name,pid,startedUtc
