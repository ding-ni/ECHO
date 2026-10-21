$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:CONDA_PKGS_DIRS = Join-Path $projectRoot '本地缓存\conda'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '本地缓存\pip'
Set-Location -LiteralPath $projectRoot
conda create --prefix (Join-Path $projectRoot '本地环境') python=3.11 pip -y --override-channels -c conda-forge
if ($LASTEXITCODE -ne 0) { throw '环境创建失败' }
$projectPython = Join-Path $projectRoot '本地环境\python.exe'
& $projectPython -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'PyTorch安装失败' }
& $projectPython -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw '项目安装失败' }
& $projectPython -m echo doctor
