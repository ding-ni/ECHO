param(
    [ValidateSet('保存并验证', '仅保存', '验证')]
    [string]$步骤 = '保存并验证'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$credentialPath = Join-Path $projectRoot '本地运行/私密认证/earthdata_token.dpapi'

if ($env:OS -ne 'Windows_NT') { throw '本脚本使用Windows当前用户DPAPI；Linux需单独配置本机认证。' }
if ($步骤 -ne '验证') {
    Write-Host '请先撤销曾发送到聊天的令牌。在这里输入新令牌，输入不会显示。'
    $secureToken = Read-Host '新的Earthdata令牌（不要输入Bearer前缀）' -AsSecureString
    if ($secureToken.Length -eq 0) { throw '未输入令牌，未修改本机凭据。' }
    $null = New-Item -ItemType Directory -Path (Split-Path $credentialPath -Parent) -Force
    $secureToken | ConvertFrom-SecureString | Set-Content -LiteralPath $credentialPath -Encoding ASCII
    $secureToken.Dispose()
    Write-Host '令牌已通过Windows当前用户DPAPI加密保存；该目录已被Git忽略。'
}
if ($步骤 -eq '仅保存') { return }
if (-not (Test-Path -LiteralPath $credentialPath)) { throw '请先运行默认步骤录入本机令牌。' }

$savedEnvironmentValue = $env:ECHO_EARTHDATA_TOKEN
$tokenHandle = [IntPtr]::Zero
$secureToken = $null
Push-Location $projectRoot
try {
    $secureToken = Get-Content -LiteralPath $credentialPath -Raw | ConvertTo-SecureString
    $tokenHandle = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    $env:ECHO_EARTHDATA_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenHandle).Trim()
    # The child receives the token in its process environment, never argv or logs.
    & (Join-Path $projectRoot '本地环境/python.exe') -m echo.data.nldas_access
    if ($LASTEXITCODE -ne 0) { throw 'NASA样例验证未通过。请核对账号应用授权、令牌有效期或网络连接。' }
}
finally {
    $env:ECHO_EARTHDATA_TOKEN = $savedEnvironmentValue
    if ($tokenHandle -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenHandle) }
    if ($null -ne $secureToken) { $secureToken.Dispose() }
    Pop-Location
}
