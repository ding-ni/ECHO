$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:CONDA_PKGS_DIRS = Join-Path $projectRoot '本地缓存\conda'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '本地缓存\pip'
$env:TEMP = Join-Path $projectRoot '本地缓存\temp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Set-Location -LiteralPath $projectRoot
conda create --prefix (Join-Path $projectRoot '本地环境') python=3.11 pip -y --override-channels -c https://repo.anaconda.com/pkgs/main
if ($LASTEXITCODE -ne 0) { throw '环境创建失败' }
$projectPython = Join-Path $projectRoot '本地环境\python.exe'
& $projectPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'PyTorch安装失败' }
& $projectPython -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw '项目安装失败' }
& $projectPython -m echo doctor
