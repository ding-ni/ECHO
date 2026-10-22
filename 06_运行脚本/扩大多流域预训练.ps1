param([switch]$WaitForPrepared,
      [string]$Index = '本地数据/模型视图/多流域Q预训练_v1/index.json',
      [string]$RunRoot = '本地运行/多流域Q预训练_v1')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = 'utf-8'
$python = Join-Path $projectRoot '本地环境/python.exe'
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$state = Join-Path $RunRoot '队列状态.json'
try {
    if ($WaitForPrepared) {
        @{status = 'waiting_for_data_preparation'; pid = $PID; updated = (Get-Date).ToString('o')} |
            ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
        while (-not (Test-Path -LiteralPath $Index)) { Start-Sleep -Seconds 5 }
    } else {
        & $python -u -m echo.data.scale prepare --manifest 本地数据/审计/候选站点_边界复核.csv `
            --roles 01_数据准备/固定水系角色_v1.json --mapping 01_数据准备/气象字段核验_首批八站_v3.json `
            --output (Split-Path -Parent $Index)
        if ($LASTEXITCODE -ne 0) { throw '多流域Q数据准备失败' }
    }
    @{status = 'training_full_and_three_folds'; pid = $PID; updated = (Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
    & $python -u -m echo.training.multibasin --index $Index --config 02_模型训练/多流域HBV预训练.yaml `
        --output (Join-Path $RunRoot '区域水文')
    if ($LASTEXITCODE -ne 0) { throw '多流域训练失败；保留检查点，重启同一命令将恢复' }
    @{status = 'source_Q_pretraining_complete'; pid = $PID; updated = (Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
} catch {
    @{status = 'failed'; pid = $PID; error = $_.ToString(); updated = (Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
    throw
}
