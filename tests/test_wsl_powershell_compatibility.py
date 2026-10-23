"""Use the real Windows shells to catch encoding and parameter-binding regressions."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell compatibility")
@pytest.mark.parametrize("shell", ["powershell.exe", "pwsh.exe"])
def test_wsl_entrypoint_chinese_parameters_parse_in_real_shell(shell):
    executable = shutil.which(shell)
    if not executable:
        pytest.skip(f"{shell} is not installed")
    environment = dict(os.environ)
    environment["ECHO_WSL_TEST_SCRIPT"] = str(
        Path(__file__).resolve().parents[1] / "06_运行脚本" / "准备WSL2完整模型环境.ps1"
    )
    environment["ECHO_WSL_TEST_PARAMETER"] = "步骤"
    command = """
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    $tokens = $null; $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $env:ECHO_WSL_TEST_SCRIPT, [ref]$tokens, [ref]$parseErrors) | Out-Null
    if ($parseErrors.Count -gt 0) { $parseErrors | Out-String | Write-Output; exit 1 }
    $metadata = Get-Command $env:ECHO_WSL_TEST_SCRIPT
    if (-not $metadata.Parameters.ContainsKey($env:ECHO_WSL_TEST_PARAMETER)) { exit 2 }
    """
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        env=environment, capture_output=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
