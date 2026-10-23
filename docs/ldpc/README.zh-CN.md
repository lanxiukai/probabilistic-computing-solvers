# LDPC最可能码字搜索

[English](README.md) | 简体中文

实测见[RTX 5090报告](experiment-report.zh-CN.md)。研究经典二元线性LDPC的错误坐标译码：
硬判决$h$、码字$c$及$e=c\oplus h$满足GF(2)上的$He=Hh$，
目标为最小化$\sum_i q_i e_i$。接收求解器只取得$H$、综合及
$q_i=\operatorname{round}(256\min(|LLR_i|,32))$。
带符号LLR、硬判决和发送真值均在求解器之外；量化似然与原始连续似然是不同目标。

源码为`src/combinatorial_solvers/ldpc/`。原生XOR因子与着色Gibbs更新原始位；
度六二次编码每校验增加两个松弛位，惩罚为$\Gamma(\sum e_i-2u-4v-\sigma)^2$。
大码原生面板有32766数据p-bit、没有辅助位；二次面板有16380数据位加16380辅助位，共32760。
独立的WiFi-648面板使用648个原始位。
副本、缓存、更新算术和经典BP状态分别记账。

有限惩罚权重不证明全局等价ML优化，只比较真正可行物理态的原始似然。
联合松弛位更新及校验惩罚渐变作为明确变体；BP暖启动须报告自身成本，
以及最终状态是否改善初始译码输出。保留同GPU min-sum／sum-product及外部BP／OSD／LSD。
OSD排序必须使用完整log-odds，而不是单侧代价近似。

独立检查全部综合、辅助残差、原始成本和重建码字；可行性、ML证据、FER／BER及耗时分别报告：
即使精确ML码字也可能不同于发送字。采用声明的$E_s/N_0$，
不能从未核验设计码率推断$E_b/N_0$；全零发送须有已检查的规范对称性。
大码每组八帧及重复采样种子不能证明罕见错误率或大码ML最优性。
保留此前OSD超时限制，并将OSD约束在声明的小码范围。

按[通用步骤](../reproduction.zh-CN.md)运行`ldpc`。
示例创建合成码且只保存接收输入；作者库及WiFi矩阵来源见[外部基线](../external-baselines.zh-CN.md)。

## 参考来源

以下保留专题研究中使用的来源入口，包括背景和未实施对照。引用不代表复现，也不代表本次重新阅读全文。基线固定版本见外部基线清单。

- R1: [Error-Rate Reduction in LDPC Decoding via Bit-Aligned Temporal Reinforcement in Parallel Probabilistic-Bit Dynamics](https://arxiv.org/html/2609.12389v1)
- R2: [代码仓库的已读版本](https://github.com/nonizawa/pSA_LDPC/blob/99bde4cd173ce90bdde098873925589fb9d4a05d/README.md)
- R3: [quantumgizmos/ldpc](https://github.com/quantumgizmos/ldpc)
- R4: [Localized statistics decoding for quantum low-density parity-check codes](https://arxiv.org/abs/2406.18655)
- R5: [Tavildar LDPC](https://github.com/tavildar/LDPC)
