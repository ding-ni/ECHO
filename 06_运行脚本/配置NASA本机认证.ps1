param(
    [ValidateSet('保存并验证', '仅保存', '验证')]
    [string]$步骤 = '保存并验证'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$credentialPath = Join-Path $projectRoot '本地运行/私密认证/earthdata_token.dpapi'
. (Join-Path $PSScriptRoot 'NASA凭据工具.ps1')

if ($env:OS -ne 'Windows_NT') { throw '本脚本使用Windows当前用户DPAPI；Linux需单独配置本机认证。' }
if ($步骤 -ne '验证') {
    Write-Host '请先撤销曾发送到聊天的令牌。在这里输入新令牌，输入不会显示。'
    $secureToken = Read-Host '新的Earthdata令牌（不要输入Bearer前缀）' -AsSecureString
    try { Save-EchoEarthdataToken -LiteralPath $credentialPath -Token $secureToken }
    finally { $secureToken.Dispose() }
    Write-Host '令牌已通过Windows当前用户DPAPI加密保存；该目录已被Git忽略。'
}
if ($步骤 -eq '仅保存') { return }
if (-not (Test-Path -LiteralPath $credentialPath)) { throw '请先运行默认步骤录入本机令牌。' }

$savedEnvironmentValue = $env:ECHO_EARTHDATA_TOKEN
$savedPythonEncoding = $env:PYTHONIOENCODING
$tokenHandle = [IntPtr]::Zero
$secureToken = $null
Push-Location $projectRoot
try {
    $secureToken = Read-EchoEarthdataToken -LiteralPath $credentialPath
    $tokenHandle = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    $env:ECHO_EARTHDATA_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenHandle).Trim()
    $env:PYTHONIOENCODING = 'utf-8'
    # The child receives the token in its process environment, never argv or logs.
    & (Join-Path $projectRoot '本地环境/python.exe') -m echo.data.nldas_access
    if ($LASTEXITCODE -ne 0) { throw 'NASA样例验证未通过。请核对账号应用授权、令牌有效期或网络连接。' }
}
finally {
    $env:ECHO_EARTHDATA_TOKEN = $savedEnvironmentValue
    $env:PYTHONIOENCODING = $savedPythonEncoding
    if ($tokenHandle -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenHandle) }
    if ($null -ne $secureToken) { $secureToken.Dispose() }
    Pop-Location
}
