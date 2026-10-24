[CmdletBinding()]
param(
    [ValidateSet('检查', '平台', '发行版')][string]$步骤 = '检查',
    [string]$安装位置 = 'F:\博士课题研究\ECHO_WSL\Ubuntu2404',
    [ValidateSet('官方安装包','WSL在线目录')][string]$安装源 = '官方安装包',
    [switch]$仅预览
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$distribution = 'Ubuntu-24.04'

function Read-WslOutput {
    param([ValidateSet('--status','--help')][string]$Argument)
    # The not-yet-installed Windows shim emits UTF-16LE through a redirected pipe.
    $info = [Diagnostics.ProcessStartInfo]::new('wsl.exe')
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    # .NET Framework in Windows PowerShell 5.1 has no ArgumentList property.
    # Only the two literal read-only switches above are allowed here.
    $info.Arguments = $Argument
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    $outBytes = [IO.MemoryStream]::new()
    $errBytes = [IO.MemoryStream]::new()
    try {
        $process.Start() | Out-Null
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($outBytes)
        $errTask = $process.StandardError.BaseStream.CopyToAsync($errBytes)
        $process.WaitForExit()
        $outTask.GetAwaiter().GetResult()
        $errTask.GetAwaiter().GetResult()
        $parts = foreach ($buffer in @($outBytes, $errBytes)) {
            $bytes = $buffer.ToArray()
            $nulls = @($bytes | Where-Object { $_ -eq 0 }).Count
            $encoding = if ($nulls -gt ($bytes.Length / 10)) { [Text.Encoding]::Unicode } else { [Text.Encoding]::UTF8 }
            $encoding.GetString($bytes).Trim([char]0xFEFF)
        }
        return [pscustomobject]@{ ExitCode = $process.ExitCode; Text = ($parts -join "`n").Trim() }
    }
    finally { $outBytes.Dispose(); $errBytes.Dispose(); $process.Dispose() }
}

function Invoke-WslChecked {
    param([string[]]$Arguments)
    & wsl.exe @Arguments
    if ($LASTEXITCODE -notin @(0, 3010)) {
        throw "WSL 命令失败（退出码 $LASTEXITCODE）；请保留屏幕信息，不继续后续安装。"
    }
}

function Assert-WslLocationCapability {
    param($HelpResult)
    # WSL 3.0.3 returns -1 even when --help successfully prints every option.
    # A declared option is the capability evidence; help's exit code alone is not.
    if ($HelpResult.Text -notmatch '(?m)^\s*--location(?:\s|,|$)') {
        throw "WSL 帮助未声明 --location（帮助退出码 $($HelpResult.ExitCode)）。请先核对 wsl --version 与 wsl --help；此检查没有启动安装。"
    }
}

if ($仅预览 -and $步骤 -ne '发行版') { throw '-仅预览只适用于 -步骤 发行版；默认检查步骤本来就是只读。' }

if ($步骤 -eq '检查') {
    $os = Get-CimInstance Win32_OperatingSystem
    $computer = Get-CimInstance Win32_ComputerSystem
    $processor = Get-CimInstance Win32_Processor | Select-Object -First 1
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    $drives = @(Get-PSDrive -PSProvider FileSystem | Where-Object Name -in @('C','F') |
        Select-Object Name, Free, Used)
    $wslStatus = Read-WslOutput -Argument '--status'
    $result = [ordered]@{
        checked_utc = [DateTime]::UtcNow.ToString('o')
        windows = $os.Caption
        windows_build = $os.BuildNumber
        memory_bytes = $computer.TotalPhysicalMemory
        virtualization_firmware_enabled = $processor.VirtualizationFirmwareEnabled
        hypervisor_present = $computer.HypervisorPresent
        elevated = $admin
        drives = $drives
        planned_distribution = $distribution
        planned_install_location = $安装位置
        changes_system = $false
        wsl_status_exit_code = $wslStatus.ExitCode
        wsl_status_text = $wslStatus.Text
    }
    $outDir = Join-Path $projectRoot '本地运行/WSL2准备'
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outDir 'Windows前置检查.json') -Encoding utf8
    $result | ConvertTo-Json -Depth 6 | Write-Output
    Write-Host '本步骤只检查。安装平台：在管理员 PowerShell 运行本脚本 -步骤 平台。'
    return
}

if ($步骤 -eq '平台') {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) { throw 'Windows 启用系统组件需要管理员权限。请以管理员身份打开 PowerShell 后运行此步骤。' }
    foreach ($feature in @('VirtualMachinePlatform','Microsoft-Windows-Subsystem-Linux')) {
        $state = Get-WindowsOptionalFeature -Online -FeatureName $feature
        if ($state.State -ne 'Enabled') {
            Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart | Out-Host
        }
    }
    Invoke-WslChecked -Arguments @('--install','--no-distribution','--web-download')
    Write-Host '平台安装命令已完成；尚未安装 Ubuntu。本脚本不执行重启。'
    Write-Host '如 Windows 提示需要重启，请先安排下载任务断点，再手动重启；之后运行 -步骤 发行版。'
    return
}

$resolvedLocation = [IO.Path]::GetFullPath($安装位置)
if ([IO.Path]::GetPathRoot($resolvedLocation) -ne 'F:\') { throw '本机 C 盘空间不足，本脚本要求发行版安装在 F 盘。' }
$helpResult = Read-WslOutput -Argument '--help'
Assert-WslLocationCapability -HelpResult $helpResult
$registered = @()
$registryRoot = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
if (Test-Path -LiteralPath $registryRoot) {
    $registered = @(Get-ChildItem -LiteralPath $registryRoot | Get-ItemProperty |
        Where-Object DistributionName -eq $distribution)
}
if ($registered.Count -gt 0) {
    $registered | Select-Object DistributionName, Version, BasePath | Format-List | Out-Host
    if ($registered[0].Version -ne 2) { throw '已有同名发行版不是 WSL2；请先核查，脚本不会自动迁移或删除。' }
    Write-Host '已有同名 WSL2 发行版，复用该环境。若 BasePath 不在 F 盘，请先检查空间再安装模型依赖。'
    return
}
if ((Test-Path -LiteralPath $resolvedLocation) -and
    @(Get-ChildItem -LiteralPath $resolvedLocation -Force).Count -gt 0) {
    throw '指定安装目录非空。请使用新的空目录，脚本不会覆盖或移动既有文件。'
}
$drive = Get-PSDrive -Name F
if ($drive.Free -lt 80GB) { throw 'F 盘可用空间低于环境预留的 80 GiB，请先检查容量。' }
$installArguments = @('--install','-d',$distribution,'--location',$resolvedLocation,'--web-download','--no-launch')
$packageSpec = $null
if ($安装源 -eq '官方安装包') {
    if ($helpResult.Text -notmatch '(?m)^\s*--from-file(?:\s|,|$)') {
        throw '当前帮助未声明 --from-file，请核对WSL版本；尚未启动下载或安装。'
    }
    $manifestPath = Join-Path $projectRoot '07_环境配置/Ubuntu2404官方安装包.json'
    $packageSpec = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($packageSpec.distribution -ne $distribution -or $packageSpec.architecture -ne 'amd64') {
        throw '固定安装包清单与Ubuntu24.04 amd64不一致。'
    }
    $cacheRoot = Join-Path $projectRoot '本地缓存/WSL'
    $packagePath = Join-Path $cacheRoot $packageSpec.filename
    $installArguments = @('--install','--from-file',$packagePath,'--name',$distribution,
        '--location',$resolvedLocation,'--version','2','--no-launch')
}
if ($仅预览) {
    [ordered]@{ distribution=$distribution; install_location=$resolvedLocation; command='wsl.exe';
        arguments=$installArguments; wsl_help_exit_code=$helpResult.ExitCode;
        location_option_verified=$true; install_source=$安装源; package=$packageSpec;
        changes_system=$false } | ConvertTo-Json -Depth 5 | Write-Output
    return
}
if ($安装源 -eq '官方安装包') {
    $python = Join-Path $projectRoot '本地环境/python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw '固定安装包下载需要项目已有的Windows Python环境。' }
    # Reuse the project's resumable SHA-256 downloader; this is not a Zenodo asset.
    $downloadCode = @'
import json, sys
from pathlib import Path
from filelock import FileLock
from echo.data.download import fetch
spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
with FileLock(str(root / '.ubuntu-package.lock'), timeout=0):
    fetch(spec, root, snapshot=False)
'@
    & $python -c $downloadCode $manifestPath $cacheRoot
    if ($LASTEXITCODE -ne 0) { throw '官方Ubuntu安装包下载或校验未完成；保留.part以便重试，未启动安装。' }
    # Avoid depending on Get-FileHash module autoload when 5.1 inherits a PowerShell 7 environment.
    $hasher = [Security.Cryptography.SHA256]::Create()
    $packageStream = [IO.File]::OpenRead($packagePath)
    try { $actualHash = [BitConverter]::ToString($hasher.ComputeHash($packageStream)).Replace('-', '').ToLowerInvariant() }
    finally { $packageStream.Dispose(); $hasher.Dispose() }
    if ($actualHash -ne $packageSpec.sha256) {
        throw 'Ubuntu安装包SHA-256不一致，未启动安装。'
    }
}
$parent = Split-Path $resolvedLocation -Parent
New-Item -ItemType Directory -Path $parent -Force | Out-Null
Invoke-WslChecked -Arguments $installArguments
Write-Host "发行版安装命令已完成。请运行：wsl -d $distribution，完成首次 Linux 用户设置。"
Write-Host '随后在 Ubuntu 内执行文档列出的 Linux 初始化命令。本脚本不会自动启动、关机或重启。'
