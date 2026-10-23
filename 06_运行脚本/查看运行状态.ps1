$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent $PSScriptRoot
Get-Content -LiteralPath (Join-Path $projectRoot '本地运行\阶段状态.json') -Encoding utf8
Get-ChildItem -LiteralPath (Join-Path $projectRoot '本地数据\原始') -Recurse -Filter '*.status.json' -File | ForEach-Object {
    Write-Output $_.FullName
    Get-Content -LiteralPath $_.FullName -Encoding utf8
}
$echoRoot = Join-Path $projectRoot '本地运行/扩展水位训练_v2'
foreach ($name in @('ECHO扩容队列状态.json', 'ECHO扩容训练标准输出.log', 'ECHO扩容训练错误输出.log',
                     'ECHO联合推断_本地小样例_v1/history.json', 'ECHO联合推断_本地小样例_v1/联合逻辑验证结果.json')) {
    $path = Join-Path $echoRoot $name
    if (Test-Path -LiteralPath $path) {
        Write-Output "ECHO当前记录：$name"
        Get-Content -LiteralPath $path -Encoding utf8 -Tail 20
    }
}
Get-Content -LiteralPath (Join-Path $projectRoot '本地运行\环境与测试\执行状态.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $projectRoot '本地运行\后台任务.json') -Encoding utf8
foreach ($run in @('多流域Q预训练_v1', '扩展水位训练_v2')) {
    $runRoot = Join-Path $projectRoot "本地运行/$run"
    Write-Output "当前扩容训练：$run"
    foreach ($file in @('队列状态.json', '训练标准输出.log', '训练错误输出.log')) {
        $path = Join-Path $runRoot $file
        if (Test-Path -LiteralPath $path) { Get-Content -LiteralPath $path -Encoding utf8 -Tail 10 }
    }
    $hydro = Join-Path $runRoot '区域水文'
    if (Test-Path -LiteralPath $hydro) {
        Get-ChildItem -LiteralPath $hydro -Filter '*.status.json' -File | ForEach-Object {
            Get-Content -LiteralPath $_.FullName -Encoding utf8
        }
    }
}
