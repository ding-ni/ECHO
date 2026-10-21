# 模型评估

预测器不读取评分标签。运行顺序：

```powershell
echo predict --index 本地数据/模型视图/index.json --hydro-run 本地运行/区域水文 --baseline local_spline --output 本地运行/样条预测
echo evaluate --predictions 本地运行/样条预测 --labels 本地数据/模型视图/评分标签 --output 本地运行/样条评估
echo report --evaluation 本地运行/样条评估 --predictions 本地运行/样条预测 --labels 本地数据/模型视图/评分标签
```

主终点：事件内RMSE、站内平均、跨站等权。置信区间以终端水系整块重采样。
事件阈值固定为许可支持H的95%分位；高水位命中间隔24小时内合并，窗口前延24小时、后延48小时。
整年MAE/RMSE同时报告。168小时评分块有效H低于90%时排除该块，并报告有效小时数。
预测区间只表示当前成员分歧；尚未包含完整天气集合或校准残差，不作为总预报区间。
