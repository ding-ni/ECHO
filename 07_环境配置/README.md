# 环境配置

本地使用项目内`本地环境`（Python3.11），通过绝对路径python.exe或conda activate激活。
Windows运行`06_运行脚本/建立环境.ps1`。项目环境、pip和Conda缓存均在F盘，避免占用C盘。

本机CUDA版PyTorch以官方2.8.0+cu126为起点。安装成功后运行`echo doctor`，只有CUDA前向反向检查通过才标记GPU可用。
实际版本锁文件在完成安装与测试后导出；未通过的环境不能写成已锁定。

Linux服务器先安装与驱动相容的官方PyTorch，再`pip install -e '.[dev]'`。Mamba-3的官方提交、CUDA依赖和长序列吞吐另行实测。
项目开发模型支持CPU；空间模块与正式骨干未接入时明确报错，不静默替换。
