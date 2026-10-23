# 图着色与列表着色

[English](README.md)

保持one-hot与颜色列表约束的GPU模拟p-bit条件块热浴，配合可选团对称性约简及温度交换，用于固定颜色数的冲突最小化。

最终确认包含56次原始调用及4次回放。18次热浴中14次达到零冲突，两个25色Leighton输入均保留残差；部分路线优于匹配plateau，但CPU Tabu在5色图上更快，不能声称普遍热浴优势。最大成功存储编码46,656位，其中28,026位连接原始冲突因子，副本数另计。

- [模型、结果、资源及完整参考文献](experiment-report.zh-CN.md)
- [源码与复现指南](source-guide.zh-CN.md)
- [确认台账](../../data/graph-coloring/rtx5090/calls.json)

团约简和温度交换属于经典支撑；Tabu是独立经典参照。本实现不报告物理p-bit器件测量。
