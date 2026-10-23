# Windows当前用户DPAPI；不输出明文或密文。
function Save-EchoEarthdataToken {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [Parameter(Mandatory)][Security.SecureString]$Token
    )
    if ($Token.Length -eq 0) { throw '未输入令牌，未修改本机凭据。' }
    $null = New-Item -ItemType Directory -Path (Split-Path $LiteralPath -Parent) -Force
    $encryptedValue = ConvertFrom-SecureString -SecureString $Token
    Set-Content -LiteralPath $LiteralPath -Value $encryptedValue -Encoding ASCII -NoNewline
}

function Read-EchoEarthdataToken {
    param([Parameter(Mandatory)][string]$LiteralPath)
    # 兼容旧版Set-Content自动加入的CRLF和文本文件BOM。
    $encryptedValue = (Get-Content -LiteralPath $LiteralPath -Raw).Trim().TrimStart([char]0xFEFF).Trim()
    if ([string]::IsNullOrWhiteSpace($encryptedValue) -or $encryptedValue -match '[^0-9a-fA-F]') {
        throw '本机加密凭据文件格式异常；请用配置脚本重新录入，不要发送文件内容。'
    }
    try { ConvertTo-SecureString -String $encryptedValue -ErrorAction Stop }
    catch { throw '无法解密本机凭据；请确认当前Windows用户，或用配置脚本重新录入。' }
}
