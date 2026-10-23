#!/usr/bin/env bash
set -euo pipefail

# Run as the normal Ubuntu user. System mode asks sudo locally; no passwords in arguments.
mode="${1:-help}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$mode" == help ]]; then
    printf '%s\n' 'Usage: bash 初始化Linux完整模型环境.sh system|python|verify' \
        'system: Ubuntu 24.04 system packages and NVIDIA CUDA Toolkit 12.6 (sudo)' \
        'python: isolated Python 3.11 / PyTorch / pinned official Mamba (normal user)' \
        'verify: actual official 6-block CUDA forward/backward; save runtime evidence'
    exit 0
fi
if [[ "$mode" != system && "$mode" != python && "$mode" != verify ]]; then
    printf 'Unknown mode: %s\n' "$mode" >&2; exit 2
fi
[[ "$(uname -s)" == Linux ]] || { echo 'Linux required.' >&2; exit 1; }
[[ "$EUID" -ne 0 ]] || { echo 'Use your normal Linux user, not root; system mode uses sudo.' >&2; exit 1; }
if [[ "$HOME" == /mnt/* ]]; then
    echo 'Linux HOME/venv must be in the Linux filesystem, not a mounted Windows directory.' >&2; exit 1
fi
export PIP_CACHE_DIR="$HOME/.cache/echo/pip"
export UV_CACHE_DIR="$HOME/.cache/echo/uv"
export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/echo/python"
export TRITON_CACHE_DIR="$HOME/.cache/echo/triton"
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export OMP_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:/usr/lib/wsl/lib:$PATH"
venv="$HOME/.venvs/echo-full"
run_dir="$project_root/本地运行/WSL2准备/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$run_dir"
trap 'status=$?; printf "{\"mode\":\"%s\",\"exit_code\":%s,\"scope\":\"environment_command_only\"}\n" "$mode" "$status" > "$run_dir/command_exit.json"' EXIT

if [[ "$mode" == system ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    [[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || {
        echo 'This system recipe is verified only for Ubuntu 24.04.' >&2; exit 1;
    }
    [[ "$(uname -m)" == x86_64 ]] || { echo 'x86_64 required.' >&2; exit 1; }
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential cmake ninja-build python3 python3-venv
    keyring="$run_dir/cuda-keyring_1.1-1_all.deb"
    curl --fail --location --retry 3 --connect-timeout 20 \
        https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
        --output "$keyring"
    printf 'eea6cc5f0eaeb99082d054b8c05ae206a378e31e88048df0310d59f651dceed2  %s\n' "$keyring" | sha256sum --check
    sudo dpkg -i "$keyring"
    sudo apt-get update
    # Toolkit-only package: never install Linux NVIDIA display drivers inside WSL.
    sudo apt-get install -y --no-install-recommends cuda-toolkit-12-6
    "$CUDA_HOME/bin/nvcc" --version | tee "$run_dir/nvcc.txt"
    echo 'System prerequisites installed. Next run python mode as the same normal user.'
    exit 0
fi

if [[ "$mode" == python ]]; then
    command -v nvcc >/dev/null || { echo 'Run system mode first; Linux nvcc is required.' >&2; exit 1; }
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$run_dir/gpu.txt"
    if [[ ! -x "$HOME/.venvs/echo-bootstrap/bin/python" ]]; then
        python3 -m venv "$HOME/.venvs/echo-bootstrap"
    fi
    bootstrap="$HOME/.venvs/echo-bootstrap/bin/python"
    "$bootstrap" -m pip install 'uv>=0.9,<1'
    "$bootstrap" -m uv python install 3.11
    if [[ ! -x "$venv/bin/python" ]]; then
        "$bootstrap" -m uv venv --seed --python 3.11 "$venv"
    fi
    "$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version'
    "$venv/bin/python" -m pip install --upgrade pip setuptools wheel packaging ninja
    "$venv/bin/python" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu126
    "$venv/bin/python" -m pip install -c "$project_root/07_环境配置/linux完整模型约束.txt" \
        -e "$project_root[dev,spatial]"
    "$venv/bin/python" -m pip install --no-build-isolation \
        -c "$project_root/07_环境配置/linux完整模型约束.txt" \
        -r "$project_root/07_环境配置/官方Mamba3固定依赖.txt"
    "$venv/bin/python" -m pip check
    "$venv/bin/python" -m pip freeze > "$run_dir/installed_candidate.txt"
    echo 'Packages installed; official CUDA acceptance has NOT run yet. Next run verify mode.'
    exit 0
fi

[[ -x "$venv/bin/python" ]] || { echo 'Run python mode first.' >&2; exit 1; }
"$venv/bin/python" -m pip check
cd "$project_root"
"$venv/bin/python" -m echo.training.protocol_doctor --exercise --output "$run_dir/官方Mamba3运行验收.json"
"$venv/bin/python" -m pip freeze > "$run_dir/linux实测依赖锁.txt"
git rev-parse HEAD > "$run_dir/code_commit.txt"
echo "Official backbone CUDA acceptance passed. Evidence: $run_dir"
echo 'This checks the backbone only; real spatial inputs and complete A-E training are still required.'
