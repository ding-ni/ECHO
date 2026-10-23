# 完整主模型：本机 WSL2 与 RTX 3060

2026-10-24用户已选择本机WSL2，共用笔记本GPU。当前实测Windows 11家庭中文版25H2、虚拟化已开启、RTX 3060 Laptop 6GB、驱动610.62、内存16GB。WSL尚未安装，当前代理终端没有Windows管理员令牌；尚未启用系统组件、安装发行版或执行重启。

C盘剩余约18GiB，F盘约3.18TiB。Ubuntu的VHDX放在`F:\博士课题研究\ECHO_WSL\Ubuntu2404`，Python环境、编译及包缓存在该Linux文件系统中；原始资料继续复用F盘项目目录。WSL程序本身仍可能占用少量系统盘空间。WSL提供Linux运行环境，使用的仍是同一块6GB GPU。

## 1. Windows只读检查（现在可运行）

```powershell
cd 'F:\博士课题研究\ECHO'
.\06_运行脚本\准备WSL2完整模型环境.ps1
```

结果写入忽略目录`本地运行/WSL2准备/Windows前置检查.json`；脚本默认不安装任何系统组件。

## 2. Windows平台（管理员PowerShell）

入口脚本以UTF-8 BOM保存，兼容Windows PowerShell5.1和PowerShell7。中文乱码解析错误的修复已在两个版本中通过语法、中文参数识别和实际只读检查；平台安装仍需要管理员窗口执行。

Windows启用系统组件需要真正的管理员令牌，代理工具的沙箱授权不提供该权限。在“以管理员身份运行”的PowerShell执行：

```powershell
& 'F:\博士课题研究\ECHO\06_运行脚本\准备WSL2完整模型环境.ps1' -步骤 平台
```

脚本启用虚拟机平台、WSL组件并安装WSL软件，不安装默认发行版，不自动重启。如果Windows提示需要重启，请先安排正在运行的NASA/DEM下载作业，再手动重启；下载器保留已校验文件及续传状态。不要为了安装环境同时启动第二个下载写入进程。

## 3. F盘安装Ubuntu 24.04

平台安装和所需重启完成后，以正常使用项目的Windows账户执行（不要换成另一名管理员账户）：

```powershell
cd 'F:\博士课题研究\ECHO'
.\06_运行脚本\准备WSL2完整模型环境.ps1 -步骤 发行版
wsl -l -v
wsl -d Ubuntu-24.04
```

脚本使用Microsoft已公开的`--location`、`--web-download`、`--no-launch`选项，拒绝覆盖非空目录。若当前版本缺少`--location`，先`wsl --update`，不要改用占满C盘的默认安装。新安装默认WSL2；请在`wsl -l -v`中确认VERSION=2。首次进入Ubuntu时自行创建普通Linux用户名、密码，无需发送到聊天。

可选资源模板见[WSL资源配置示例](WSL资源配置示例.wslconfig)：8GB内存、4核、4GB交换文件在F盘。它对全部WSL2发行版生效，需核对已有`%UserProfile%\.wslconfig`后手动合并；脚本不会覆盖全局设置，也不会执行`wsl --shutdown`。模板生效通常需要全部WSL实例停止后重新启动。

## 4. Ubuntu内安装完整模型候选环境

下面命令都在Ubuntu终端运行，以普通Linux用户执行；`system`步骤的sudo密码只在本机终端输入。

```bash
cd '/mnt/f/博士课题研究/ECHO'
bash '06_运行脚本/初始化Linux完整模型环境.sh' system
bash '06_运行脚本/初始化Linux完整模型环境.sh' python
bash '06_运行脚本/初始化Linux完整模型环境.sh' verify
```

脚本依次执行：

- 安装Ubuntu编译工具和NVIDIA官方WSL源的`cuda-toolkit-12-6`，核对源安装包SHA-256；不安装Linux NVIDIA显示驱动。
- 在`~/.venvs/echo-full`创建独立Python3.11环境，安装PyTorch2.9.1+cu126、项目、固定提交的官方Mamba-3。Windows`本地环境/python.exe`保持现状。
- 实际执行6块、宽256、MIMO4的官方历史编码器CUDA前向和反向，检查有限值及每块参数梯度；不改成其他SSM/LSTM。通过后保存实际依赖锁、代码提交号、GPU和峰值显存。

**Linux不能照搬原先PyTorch2.8的组合**：固定Mamba提交的setup.py要求Triton≥3.5、TileLang=0.1.8、apache-tvm-ffi≤0.1.12。已从官方wheel元数据核验PyTorch2.9.1+cu126依赖Triton3.5.1，约束写在[Linux完整模型约束](linux完整模型约束.txt)。这些是待验收的安装候选，尚不是已通过的环境锁；解析或GPU执行失败时保留失败原因，不静默升级或替换骨干。

安装可能需要下载数GB依赖，编译并发限制为1。脚本可以按同一步骤重新运行，已有虚拟环境会复用，不删除环境或清空缓存。缓存和Python均在Linux HOME内；HOME应位于F盘VHDX中的Linux文件系统。

## 5. 验收与后续真实训练

每次脚本运行在`本地运行/WSL2准备/<UTC时间>-<进程号>/`留下命令结果。只有`官方Mamba3运行验收.json`内`official_backbone_forward_backward_passed=true`才表示该骨干已在本机GPU通过；失败也保存JSON并返回非零状态。

骨干检查是合成运行检查。完整真实空间资料、真实日历的episode与A—E五阶段仍需分别通过，当前没有全套真实输入齐备的站点，不把环境安装成功记作研究阶段验收。数据入口和训练参数见[完整主模型数据清单说明](../02_模型训练/完整主模型数据清单说明.md)。

今后在Linux运行时，先配置CUDA路径并激活Linux环境：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:/usr/lib/wsl/lib:$PATH"
source "$HOME/.venvs/echo-full/bin/activate"
cd '/mnt/f/博士课题研究/ECHO'
python -m echo.training.protocol_pipeline --help
```

新清单使用相对清单文件的路径；旧Windows绝对路径、WindowsPython和DPAPI凭据不能直接迁移。气象下载可继续留在Windows，Linux训练直接读取已核验资料，无需再次下载，也无需复制认证文件。

如果安装平台后手动重启了Windows，可在确认原作业已停止后恢复下载：

```powershell
cd 'F:\博士课题研究\ECHO'
.\06_运行脚本\下载原网格历史气象.ps1 -开始 '2011-10-01T00:00:00Z' -结束 '2021-10-01T00:00:00Z'
# DEM另开一个PowerShell终端恢复；不要与已有DEM作业同时执行。
.\06_运行脚本\准备完整模型空间资料.ps1 -DownloadDEM
```

## 官方依据

- [Microsoft安装WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Microsoft命令选项及安装位置](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [固定Mamba提交的安装依赖](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/setup.py)
- [PyTorch官方CUDA12.6 wheel索引](https://download.pytorch.org/whl/cu126/torch/)

本次官方页面、setup.py与wheel元数据快照保留在忽略目录`本地参考资料/sources/wsl_setup_20261024/`。
