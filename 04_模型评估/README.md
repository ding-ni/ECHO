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
旧开发版预测区间只表示成员分歧。完整主链分别输出`predictions_member_only`（联合代码分歧）与`predictions`（加来源AR(1)残差的总成员）索引；两者用同一评分器分别评价，按运行记录解释。它们仍不自动包含真实天气集合误差。
## 固定洪水案例对照

评分侧可单独冻结案例清单。按站号顺序，每站取满足原评分覆盖门槛的前两个事件；阈值为支持H的95%分位，24小时合并、前24/后48小时。选例函数不接收预测或误差，已有清单禁止覆盖。本轮为八站开发诊断，不能称为前瞻预注册或完整分层确认性案例。

```powershell
python -m echo.evaluation.cases freeze --index 本地数据/模型视图/首批八站_v2/index.json --labels 本地数据/模型视图/首批八站_v2/评分标签 --manifest 本地运行/真实首批八站_v2/模型比较/固定洪水案例.json
python -m echo.evaluation.cases render --manifest 本地运行/真实首批八站_v2/模型比较/固定洪水案例.json --labels 本地数据/模型视图/首批八站_v2/评分标签 --predictions Persistence=本地运行/真实持续性_八站_v2/预测 HBV-Spline=本地运行/真实地方样条_八站_v2/预测 ECHO-Static=本地运行/真实首批八站_v2/预测/echo_static Direct-LSTM=本地运行/真实首批八站_v2/预测/direct_lstm ECHO-Dynamic=本地运行/真实首批八站_v2/预测/echo_dynamic --output 本地运行/真实首批八站_v2/模型比较/固定案例
```

绘图要求各模型有相同数据索引、支持H、未来时间和完整年预算；发现失败或不一致时拒绝绘图。逐时观测、预测和图留在忽略的本地运行目录。
