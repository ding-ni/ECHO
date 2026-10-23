param(
    [string]$Config = "02_模型训练/最终架构_本地小样本.yaml",
    [string]$Output = "本地运行/正式架构_本地小样本_v1"
)

$ErrorActionPreference = "Stop"
$python = "本地环境/python.exe"
$index = "本地数据/模型视图/扩展水位训练_v2/index.json"
$spatial = "本地数据/模型视图/正式空间输入_小样本_v6/index.json"

& $python -m echo.training.formal_pipeline `
    --index $index `
    --spatial-index $spatial `
    --config $Config `
    --output $Output
if ($LASTEXITCODE -ne 0) {
    throw "正式架构小样本运行失败，退出码 $LASTEXITCODE"
}
