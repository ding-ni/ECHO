param(
    [string]$Index = '本地数据/模型视图/首批八站_v2/index.json',
    [string]$RunRoot = '本地运行/真实首批八站_v2',
    [ValidateSet('hydrology', 'baselines', 'echo_static', 'direct_lstm', 'echo_dynamic', 'echo_gaussian')]
    [string[]]$Stages = @('hydrology', 'baselines')
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '本地环境/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$labels = Join-Path (Split-Path -Parent $Index) '评分标签'
$hydro = Join-Path $RunRoot '区域水文'
$configs = @{
    hydrology = @('区域水文', '02_模型训练/区域水文.yaml')
    echo_static = @('静态ECHO', '02_模型训练/静态ECHO.yaml')
    direct_lstm = @('直接水位LSTM', '02_模型训练/直接水位LSTM.yaml')
    echo_dynamic = @('动态ECHO', '02_模型训练/动态ECHO.yaml')
    echo_gaussian = @('Gaussian开发版', '02_模型训练/Gaussian开发版.yaml')
}
foreach ($stage in $Stages) {
    if ($stage -ne 'baselines') {
        $entry = $configs[$stage]
        & $python -m echo train --index $Index --config $entry[1] --output (Join-Path $RunRoot $entry[0])
        if ($LASTEXITCODE -ne 0) { throw "训练失败：$stage" }
        if ($stage -eq 'hydrology') { continue }
        $models = @($stage)
    } else { $models = @('persistence', 'local_spline') }
    foreach ($model in $models) {
        foreach ($split in @('validation', 'development_test')) {
            $prediction = Join-Path $RunRoot "预测/$model/$split"
            $evaluation = Join-Path $RunRoot "评估/$model/$split"
            $arguments = @('-m', 'echo', 'predict', '--index', $Index, '--hydro-run', $hydro,
                '--split', $split, '--output', $prediction)
            if ($stage -eq 'baselines') { $arguments += @('--baseline', $model) }
            else { $arguments += @('--model-run', (Join-Path $RunRoot $configs[$stage][0])) }
            & $python @arguments
            if ($LASTEXITCODE -ne 0) { throw "预测失败：$model/$split" }
            & $python -m echo evaluate --predictions $prediction --labels $labels --output $evaluation
            if ($LASTEXITCODE -ne 0) { throw "评分失败：$model/$split" }
            & $python -m echo report --evaluation $evaluation --predictions $prediction --labels $labels
            if ($LASTEXITCODE -ne 0) { throw "报告失败：$model/$split" }
        }
    }
}
