param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('echo_static', 'direct_lstm', 'echo_dynamic', 'echo_gaussian')]
    [string]$Model,
    [string]$Index = '本地数据/模型视图/首批八站_v2/index.json',
    [string]$RunRoot = '本地运行/真实首批八站_v2'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '本地环境/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$labels = Join-Path (Split-Path -Parent $Index) '评分标签'
$names = @{
    echo_static = '静态ECHO'
    direct_lstm = '直接水位LSTM'
    echo_dynamic = '动态ECHO'
    echo_gaussian = 'Gaussian开发版'
}
$modelRun = Join-Path $RunRoot $names[$Model]
$hydro = Join-Path $RunRoot '区域水文'
if (-not (Test-Path -LiteralPath (Join-Path $modelRun 'model.pt'))) {
    throw "模型尚未完成训练：$modelRun"
}
foreach ($split in @('validation', 'development_test')) {
    foreach ($kind in @('预测', '评估')) {
        $target = Join-Path $RunRoot "$kind/$Model/$split"
        if (Test-Path -LiteralPath $target) { throw "保留已有结果，请使用新的RunRoot：$target" }
    }
}
foreach ($split in @('validation', 'development_test')) {
    $prediction = Join-Path $RunRoot "预测/$Model/$split"
    $evaluation = Join-Path $RunRoot "评估/$Model/$split"
    & $python -m echo predict --index $Index --hydro-run $hydro --model-run $modelRun `
        --split $split --output $prediction
    if ($LASTEXITCODE -ne 0) { throw "预测失败：$Model/$split" }
    $predictionRecord = Get-Content -LiteralPath (Join-Path $prediction 'predictions.json') -Raw | ConvertFrom-Json
    if ($predictionRecord.failures.Count -gt 0 -or $predictionRecord.stations.Count -eq 0) {
        throw "预测含失败或无目标站，查看$prediction/predictions.json"
    }
    & $python -m echo evaluate --predictions $prediction --labels $labels --output $evaluation
    if ($LASTEXITCODE -ne 0) { throw "评分失败：$Model/$split" }
    $summary = Get-Content -LiteralPath (Join-Path $evaluation 'summary.json') -Raw | ConvertFrom-Json
    if ($summary.failures.Count -gt 0 -or $summary.sites -ne $predictionRecord.stations.Count) {
        throw "评分含失败，查看$evaluation/summary.json"
    }
    & $python -m echo report --evaluation $evaluation --predictions $prediction --labels $labels
    if ($LASTEXITCODE -ne 0) { throw "报告失败：$Model/$split" }
}
