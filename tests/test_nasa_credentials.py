"""Exercise the actual PowerShell DPAPI helpers using synthetic credentials."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt" or not shutil.which("pwsh"), reason="Windows DPAPI / pwsh required")
def test_dpapi_roundtrip_and_legacy_newlines(tmp_path):
    helper = Path(__file__).resolve().parents[1] / "06_运行脚本/NASA凭据工具.ps1"
    check = tmp_path / "check.ps1"
    check.write_text(
        """param([string]$Helper, [string]$Work)
$ErrorActionPreference = 'Stop'
. $Helper
$path = Join-Path $Work 'synthetic.dpapi'
$expected = 'synthetic-test-value-never-a-real-token'
$source = ConvertTo-SecureString -String $expected -AsPlainText -Force
try { Save-EchoEarthdataToken -LiteralPath $path -Token $source }
finally { $source.Dispose() }
$saved = Get-Content -LiteralPath $path -Raw
if ($saved -ne $saved.Trim()) { throw 'New files must not include trailing whitespace' }
if ($saved.Contains($expected)) { throw 'Credential was saved as plaintext' }
foreach ($value in @($saved, ($saved + "`r`n"), ("  " + $saved + "`n"))) {
    Set-Content -LiteralPath $path -Value $value -NoNewline
    $actual = Read-EchoEarthdataToken -LiteralPath $path
    $pointer = [IntPtr]::Zero
    try {
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($actual)
        if ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) -ne $expected) {
            throw 'Roundtrip mismatch'
        }
    }
    finally {
        if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
        $actual.Dispose()
    }
}
Set-Content -LiteralPath $path -Value 'not-a-valid-encrypted-value' -NoNewline
$rejected = $false
try { $unused = Read-EchoEarthdataToken -LiteralPath $path }
catch { $rejected = $true }
if (-not $rejected) { throw 'Malformed input accepted' }
Write-Output 'DPAPI roundtrip, legacy whitespace, and malformed-input rejection OK'
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(check), str(helper), str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rejection OK" in result.stdout
