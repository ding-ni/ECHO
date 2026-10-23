param(
    [string]$开始 = '2011-10-01T00:00:00Z',
    [string]$结束 = '2011-10-02T00:00:00Z',
    [int]$并发 = 2,
    [switch]$仅预览
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'NASA凭据工具.ps1')
$savedToken = $env:ECHO_EARTHDATA_TOKEN
$savedEncoding = $env:PYTHONIOENCODING
$handle = [IntPtr]::Zero
$secureToken = $null
Push-Location $projectRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $arguments = @('-m', 'echo.data.nldas_batch', '--start', $开始, '--end', $结束, '--workers', $并发)
    if (-not $仅预览) {
        $secureToken = Read-EchoEarthdataToken -LiteralPath (Join-Path $projectRoot '本地运行/私密认证/earthdata_token.dpapi')
        $handle = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        $env:ECHO_EARTHDATA_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($handle).Trim()
        $arguments += '--download'
    }
    & (Join-Path $projectRoot '本地环境/python.exe') @arguments
    if ($LASTEXITCODE -ne 0) { throw 'NLDAS下载未全部完成，可重复相同命令复核并恢复未完成小时。' }
}
finally {
    $env:ECHO_EARTHDATA_TOKEN = $savedToken
    $env:PYTHONIOENCODING = $savedEncoding
    if ($handle -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($handle) }
    if ($null -ne $secureToken) { $secureToken.Dispose() }
    Pop-Location
}
