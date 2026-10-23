"""Read-only official-backbone readiness; optional genuine CUDA forward/backward."""

import argparse
import importlib.util
import platform
import shutil

import torch

from echo.models.official_mamba import OFFICIAL_COMMIT, OFFICIAL_MODULE_SHA256, OfficialHistoryEncoder
from echo.utils import utc_now, write_json


def inspect(exercise=False):
    present = importlib.util.find_spec("mamba_ssm") is not None
    result = {"checked_utc": utc_now(), "operating_system": platform.system(),
              "python": platform.python_version(), "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
              "nvcc_on_path": shutil.which("nvcc") is not None, "official_mamba_package_present": present,
              "official_commit_required": OFFICIAL_COMMIT,
              "official_module_sha256_required": OFFICIAL_MODULE_SHA256,
              "official_backbone_forward_backward_passed": False,
              "synthetic_runtime_test_is_real_hydrological_training": False}
    if not exercise:
        return result
    if platform.system() != "Linux" or not present or not torch.cuda.is_available():
        raise RuntimeError("当前需要Linux、CUDA和固定官方Mamba-3；未替换骨干，也未执行GPU验收")
    torch.cuda.reset_peak_memory_stats()
    encoder = OfficialHistoryEncoder().cuda()
    history = torch.randn(1, 168, 8, device="cuda", requires_grad=True)
    environment = torch.randn(1, 256, device="cuda", requires_grad=True)
    output = encoder(history, environment)
    output.square().mean().backward()
    if not torch.isfinite(output).all() or not torch.isfinite(history.grad).all():
        raise RuntimeError("官方历史编码器前向或输入梯度有非有限值")
    gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("官方历史编码器参数梯度无效")
    if any(not any(p.grad is not None for p in block.parameters()) for block in encoder.blocks):
        raise RuntimeError("官方历史编码器有残差块没有参与反向传播")
    result.update(official_backbone_forward_backward_passed=True,
                  peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),
                  output_shape=list(output.shape), gradient_tensor_count=len(gradients))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = inspect(args.exercise)
    if args.output:
        write_json(args.output, result)
    print(result)
