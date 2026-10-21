# 模型训练

执行顺序：区域水文 → 静态ECHO → 动态增量 → Gaussian开发版；直接LSTM作为同预算强基线。

```powershell
echo train --index 本地数据/模型视图/index.json --config 02_模型训练/区域水文.yaml --output 本地运行/区域水文
echo train --index 本地数据/模型视图/index.json --config 02_模型训练/静态ECHO.yaml --output 本地运行/静态ECHO
```

来源Q只训练来源前端。参考Q三折按水系交叉拟合。训练元数据保存实际小时、参数量、时间、显存和提交号。
初始训练预算用于验证工程可行性，最终超参数只能由来源验证选择。
