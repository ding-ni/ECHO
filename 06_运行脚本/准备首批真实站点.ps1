param([string]$Output = '本地数据/模型视图/首批八站_v2')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '本地环境/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $python -m echo pilot-manifest `
    --manifest '本地数据/审计/候选站点_边界复核.csv' `
    --roles '01_数据准备/固定水系角色_v1.json' `
    --datum-registry '01_数据准备/历史水尺基准登记_v1.json' `
    --height-report '本地数据/审计/USGS支持期水位核验.json' `
    --source-report '本地数据/审计/USGS来源QH核验.json' `
    --output '本地数据/审计/首批试运行_v2.csv' --max-sites 10
if ($LASTEXITCODE -ne 0) { throw '站点放行失败' }
& $python -m echo prepare `
    --manifest '本地数据/审计/首批试运行_v2.csv' `
    --mapping '01_数据准备/气象字段核验_首批八站_v3.json' `
    --datum-registry '01_数据准备/历史水尺基准登记_v1.json' `
    --system-roles '01_数据准备/固定水系角色_v1.json' `
    --source-evidence '本地数据/审计/USGS来源QH核验.json' `
    --height-evidence '本地数据/审计/USGS支持期水位核验.json' `
    --height-snapshots '本地参考资料/sources/usgs_support_H' --output $Output
if ($LASTEXITCODE -ne 0) { throw '数据准备命令失败' }
$index = Get-Content -LiteralPath (Join-Path $Output 'index.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (@($index.failures).Count -ne 0) { throw '存在准备失败站点，不能继续训练' }
