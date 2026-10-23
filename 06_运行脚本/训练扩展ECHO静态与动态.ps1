$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '本地环境/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$indexQ = Join-Path $project '本地数据/模型视图/多流域Q预训练_v1/index.json'
$indexH = Join-Path $project '本地数据/模型视图/扩展水位训练_v2/index.json'
$sourceRun = Join-Path $project '本地运行/多流域Q预训练_v1/区域水文'
$runRoot = Join-Path $project '本地运行/扩展水位训练_v2'
$hydroRun = Join-Path $runRoot '区域水文_多流域'
$staticRun = Join-Path $runRoot 'ECHO静态_多流域'
$dynamicRun = Join-Path $runRoot 'ECHO动态_多流域'
$labels = Join-Path $project '本地数据/模型视图/扩展水位训练_v2/评分标签'
$queueStatus = Join-Path $runRoot 'ECHO扩容队列状态.json'

function Set-QueueStatus($status, $detail = '') {
    $record = @{ status = $status; detail = $detail; pid = $PID; updated = [DateTimeOffset]::Now.ToString('o') }
    [System.IO.File]::WriteAllText($queueStatus, ($record | ConvertTo-Json -Depth 4), [System.Text.Encoding]::UTF8)
}

function Invoke-EchoEvaluation($label, $modelRun) {
    $prediction = Join-Path $runRoot "预测/ECHO_$label/development_test"
    $score = Join-Path $runRoot "评估/ECHO_$label/development_test"
    & $python -m echo predict --index $indexH --hydro-run $hydroRun `
        --model-run $modelRun --split development_test --device cuda --output $prediction
    if ($LASTEXITCODE -ne 0) { throw "ECHO$label预测失败" }
    & $python -m echo evaluate --predictions $prediction --labels $labels --output $score
    if ($LASTEXITCODE -ne 0) { throw "ECHO$label评分失败" }
}

if (-not (Test-Path -LiteralPath $python)) { throw "项目Python环境不存在: $python" }
if (-not (Test-Path -LiteralPath (Join-Path $sourceRun 'hydro_fold2.pt'))) {
    throw '248站HBV及三折检查点尚未齐备。'
}
foreach ($path in @($staticRun, $dynamicRun)) {
    if (Test-Path -LiteralPath $path) { throw "运行目录已存在，保留已有结果并使用新编号: $path" }
}
if ((Test-Path -LiteralPath $hydroRun) -and -not (Test-Path -LiteralPath (Join-Path $hydroRun 'reference_index.json'))) {
    throw "多流域参考Q目录存在但未完成封存，请先检查: $hydroRun"
}

Push-Location $project
try {
    if (-not (Test-Path -LiteralPath $hydroRun)) {
        Set-QueueStatus 'importing_multibasin_hydrology'
        & $python -m echo import-multibasin-hydro `
            --q-index $indexQ `
            --h-index $indexH `
            --source-run $sourceRun `
            --output $hydroRun
        if ($LASTEXITCODE -ne 0) { throw '多流域HBV导入或参考Q生成失败' }
    }

    Set-QueueStatus 'auditing_crossfit_references'
    & $python -m echo reference-audit --index $indexH --hydro-run $hydroRun
    if ($LASTEXITCODE -ne 0) { throw '留折参考Q审计失败' }

    Set-QueueStatus 'training_echo_static'
    & $python -m echo train --index $indexH `
        --config (Join-Path $project '02_模型训练/扩展ECHO静态_本机GPU.yaml') `
        --output $staticRun
    if ($LASTEXITCODE -ne 0) { throw '扩展静态ECHO训练失败' }

    Set-QueueStatus 'training_echo_dynamic'
    & $python -m echo train --index $indexH `
        --config (Join-Path $project '02_模型训练/扩展ECHO动态_本机GPU.yaml') `
        --output $dynamicRun
    if ($LASTEXITCODE -ne 0) { throw '扩展动态ECHO训练失败' }

    Set-QueueStatus 'scoring_echo_static'
    Invoke-EchoEvaluation '静态' $staticRun
    Set-QueueStatus 'scoring_echo_dynamic'
    Invoke-EchoEvaluation '动态' $dynamicRun
    Set-QueueStatus 'expanded_echo_complete'
} catch {
    Set-QueueStatus 'failed' $_.Exception.Message
    throw
} finally {
    Pop-Location
}
