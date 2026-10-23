# 数据准备

原网格 NLDAS 下载需要 Earthdata 账号，见 [NASA Earthdata 注册与 NLDAS 授权指引](NASA_Earthdata注册与NLDAS授权指引.md)。

完整资料范围与冰川/积雪角色以[完整资料库定义](完整资料库定义_v1.yaml)和[最新实施对照](../00_项目说明/全套资料与完整模型实施对照_20261024.md)为准。`下载原网格历史气象.ps1`逐小时核验与恢复；`echo.geo.nldas_inputs`按真实增量多边形聚合原网格；`echo.data.environment_inventory`提取独立站点属性与历史气候，`echo.data.project_inventory`生成逐站缺项总账。站点汇总属性不会自动成为节点属性，原始瓦片不会自动成为河谷token。

先 `echo download --list`，再 `echo download`。默认五项约26GB；额外气象需要显式指定。
下载器保留固定记录、官方校验值和许可证元数据；重复运行校验已有文件并恢复未完成下载。

`echo extract --asset qh` 等命令先检查档案路径和展开体量。原始包保留。
`echo audit --root 本地数据/原始/qh --limit 20` 先输出真实字段模式。字段未核验不能进入prepare。

标准化必须提供已确认字段映射。时间支持、水位基准和终端水系未知的站点会保持待核验状态。
原始、标准化、来源训练视图、目标上下文、评分标签分开保存；未来H只由评分器读取。

`echo inventory --qh-root ... --weather-root ... --attributes-root ... --output 本地数据/审计/候选站点.csv`生成候选清单。
`echo topology --stations 本地数据/审计/候选站点.csv --output 本地数据/审计/终端水系.csv`使用官方HydroBASINS北美L12，核验点落区、NEXT_DOWN和MAIN_BAS。
拓扑归档约65MB，官方未提供的校验值不编造；记录HTTPS来源、体量、ZIP CRC及本地SHA256。
边界歧义或图链不完整的站点保持未知。水位基准和时间支持仍需独立核验，拓扑匹配不能替代这些审计。

## 下载完成后的核验

完整规则见[气象与时间核验说明](气象与时间核验说明.md)和[水位基准核验说明](水位基准核验说明.md)。以下命令均由中文脚本包装，核心实现仍在`src/echo`：

```powershell
.\06_运行脚本\继续真实数据核验.ps1 -步骤 气象核验
.\06_运行脚本\继续真实数据核验.ps1 -步骤 边界复核
.\06_运行脚本\继续真实数据核验.ps1 -步骤 水位对照
.\06_运行脚本\继续真实数据核验.ps1 -步骤 气象标准化
```

`boundary-audit`使用GAGES-II边界复核HydroBASINS终端分组；跨终端有实质重叠时隔离整个冲突水系。阈值和源文件哈希留表，不以模型误差决定排除。

`station-evidence`先固定20站审查清单，只查询USGS参数00065的过去支持期及独立站点元数据。缓存带URL、时间及SHA-256，完整重复运行复用有效缓存。整点缺测不以附近观测替代。站点海拔不作为历史水尺零点证据。

`weather-prepare`只输出气象，物理不含Q和H；它生成的`weather_index.json`不是模型训练索引。整体映射的`verified=false`和逐站`datum_verified=false`继续阻止完整prepare，气象核验完成不能绕过水位及来源Q核验。

水系归属`system_split`与站点抽样`split`分开记录：未入选的相连站点继续继承同一水系角色，不能转作目标相连水系Q来源。
