# 完整主模型的 Linux 与 WSL2 准备

当前实测为Windows、Python3.11.17、PyTorch2.8.0+cu126、RTX3060可用CUDA，官方Mamba包尚未安装。完整主模型的官方Mamba-3要求Linux CUDA/MIMO依赖；Windows现有CUDA可用不等于该骨干已经可运行。

本轮只准备配置和检查程序，没有安装WSL、改系统特性或重启电脑。可以继续在Windows完成下载、审计、空间提取、完整地理/HBV前端A阶段及独立数学性质检查。

如果采用WSL2：先按Microsoft官方WSL安装步骤准备Linux发行版和GPU透传，使用支持WSL的Windows NVIDIA驱动；按NVIDIA官方WSL CUDA文档配置，不在WSL里另装与Windows宿主驱动冲突的Linux显卡驱动。机器仍是同一块6GB GPU和16GB内存，初始batch=1、worker=0。

Linux中另建Python3.11环境；Windows `本地环境/python.exe`不能直接当LinuxPython使用。先安装与实际驱动相容的PyTorch，再安装本仓库依赖与[固定官方Mamba提交](官方Mamba3固定依赖.txt)，核验官方README要求的CUDA/TileLang/MIMO构建。依赖安装成功还需真正前向反向：

```bash
python -m echo.training.protocol_doctor
python -m echo.training.protocol_doctor --exercise --output 本地运行/官方Mamba3运行验收.json
```

检查程序核验官方模块源码SHA-256，执行6块256宽的真实历史编码器前向/反向并检查有限值及各块参数梯度；失败时不换SSM或LSTM。这个合成CUDA检查通过之后，仍要完成真实空间输入和A—E的本地小样本验证。

资料可从WSL访问`/mnt/f/博士课题研究/ECHO`。新episode清单推荐使用相对清单文件的路径；旧Windows绝对路径、DPAPI凭据文件和本地环境不能原样迁移。原网格文件和公开静态资料可复用，Linux下载认证另在该环境本机配置。

官方安装资料：

- Microsoft WSL：<https://learn.microsoft.com/windows/wsl/install>
- NVIDIA CUDA on WSL：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
- Mamba固定提交及依赖以本仓库锁文件、实际获取的官方README为准。

操作系统安装与重启方案由用户确定后实施；当前下载作业无需等待这一步。
