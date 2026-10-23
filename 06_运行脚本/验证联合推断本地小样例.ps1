$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $project
try {
    & ./本地环境/python.exe -m echo.training.local_joint `
        --index 本地数据/模型视图/扩展水位训练_v2/index.json `
        --config 02_模型训练/联合推断_本地小样例.yaml `
        --output 本地运行/扩展水位训练_v2/ECHO联合推断_本地小样例_v1
    if ($LASTEXITCODE -ne 0) { throw '联合推断小样例失败，请检查错误并保留当前结果' }
} finally {
    Pop-Location
}
