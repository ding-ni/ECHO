# 数据准备

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
