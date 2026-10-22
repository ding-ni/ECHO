param([string]$Index = '本地数据/模型视图/扩展水位训练_v2/index.json',
      [string]$RunRoot = '本地运行/扩展水位训练_v2')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '本地环境/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$state = Join-Path $RunRoot '队列状态.json'
try {
    @{status='direct_lstm_training'; pid=$PID; updated=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
    & $python -u -m echo train --index $Index --config 02_模型训练/扩展水位LSTM_加量GPU.yaml `
        --output (Join-Path $RunRoot '直接水位LSTM')
    if ($LASTEXITCODE -ne 0) { throw '扩展水位LSTM训练失败' }
    & (Join-Path $PSScriptRoot '评估已有真实模型.ps1') -Model direct_lstm -Index $Index -RunRoot $RunRoot
    $labels = Join-Path (Split-Path -Parent $Index) '评分标签'
    foreach ($split in @('validation', 'development_test')) {
        $prediction = Join-Path $RunRoot "预测/persistence/$split"
        $evaluation = Join-Path $RunRoot "评估/persistence/$split"
        & $python -m echo predict --index $Index --baseline persistence --split $split --output $prediction
        if ($LASTEXITCODE -ne 0) { throw '扩展集合持续性预测失败' }
        & $python -m echo evaluate --predictions $prediction --labels $labels --output $evaluation
        if ($LASTEXITCODE -ne 0) { throw '扩展集合持续性评分失败' }
        & $python -m echo report --evaluation $evaluation --predictions $prediction --labels $labels
        if ($LASTEXITCODE -ne 0) { throw '扩展集合持续性报告失败' }
    }
    @{status='direct_lstm_and_persistence_complete'; pid=$PID; updated=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
} catch {
    @{status='failed'; pid=$PID; error=$_.ToString(); updated=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $state -Encoding utf8
    throw
}
