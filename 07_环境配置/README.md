# 环境配置

本地使用项目内`本地环境`（Python3.11），通过绝对路径python.exe或conda activate激活。
Windows运行`06_运行脚本/建立环境.ps1`。项目环境、pip和Conda缓存均在F盘，避免占用C盘。

本机CUDA版PyTorch以官方2.8.0+cu126为起点。安装成功后运行`echo doctor`，只有CUDA前向反向检查通过才标记GPU可用。
实际版本锁文件在完成安装与测试后导出；未通过的环境不能写成已锁定。

用户已选择本机WSL2。安装入口与分步命令见[完整主模型Linux与WSL2准备](完整主模型Linux与WSL2准备.md)。Linux候选采用PyTorch2.9.1+cu126、Triton3.5.1；Windows实测版本保持2.8.0+cu126。官方Mamba固定提交已记录，GPU运行与长序列吞吐仍需实测。
项目开发模型支持CPU；空间模块与正式骨干未接入时明确报错，不静默替换。
