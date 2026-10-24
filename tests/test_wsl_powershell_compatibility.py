"""Use the real Windows shells to catch encoding and parameter-binding regressions."""

import json
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shell compatibility")
@pytest.mark.parametrize("shell", ["powershell.exe", "pwsh.exe"])
@pytest.mark.parametrize("exit_code, help_text, expected", [
    (-1, "    --install [Distro] [Options...]\n            --location <Location>\n", True),
    (0, "            --location <Location>\n", True),
    (0, "            --web-download\n", False),
    (-1, "Error: failed to query --location", False),
])
def test_location_detection_accepts_real_help_despite_nonzero_exit(shell, exit_code, help_text, expected):
    executable = shutil.which(shell)
    if not executable:
        pytest.skip(f"{shell} is not installed")
    environment = dict(os.environ)
    environment["ECHO_WSL_TEST_SCRIPT"] = str(
        Path(__file__).resolve().parents[1] / "06_运行脚本" / "准备WSL2完整模型环境.ps1"
    )
    environment["ECHO_WSL_TEST_HELP"] = json.dumps({"ExitCode": exit_code, "Text": help_text})
    environment["ECHO_WSL_TEST_EXPECTED"] = str(expected).lower()
    # Extract only the capability check: never execute the install entry point in a test.
    command = """
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    $tokens=$null; $errors=$null
    $ast=[System.Management.Automation.Language.Parser]::ParseFile(
        $env:ECHO_WSL_TEST_SCRIPT,[ref]$tokens,[ref]$errors)
    if ($errors.Count -gt 0) { exit 1 }
    $definition=$ast.Find({param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Assert-WslLocationCapability'},$true)
    if (-not $definition) { exit 2 }
    . ([scriptblock]::Create($definition.Extent.Text))
    try { Assert-WslLocationCapability -HelpResult ($env:ECHO_WSL_TEST_HELP | ConvertFrom-Json); $accepted=$true }
    catch { $accepted=$false }
    if ($accepted -ne [bool]::Parse($env:ECHO_WSL_TEST_EXPECTED)) { exit 3 }
    """
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", command],
        env=environment, capture_output=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
