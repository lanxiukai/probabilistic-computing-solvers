# RTX 5090 LDPC综合域似然搜索

[English](experiment-report.md) | 简体中文

冻结D2包含**41次自然调用、六次工作量回放、五个码和96个唯一信道帧**。
其中WiFi-648有64帧，每个大码有八帧；重复采样种子复用这些帧，
不能把独立信道分母翻倍。

NVIDIA GeForce RTX 5090 (32607 MiB); AMD Eng Sample: 100-000001537-04 (quota 17.56 vCPU); Python 3.14.6; PyTorch 2.13.0+cu130; CUDA 13.0; driver 580.178.04.

本轮设备确认复用此前已测的码、信道帧和采样种子。
帧错误按译码结果与发送字是否不同计数；包括辅助约束在内的物理模型合法性另外核对。
随机调用每帧使用四个副本，冻结`max_steps`分别为WiFi的4096、大码原生的5632、
二次模型的6144，并设600秒时间保护。这些是固定工作量上限，不是所有译码器的
统一时间预算。OSD只在WiFi-648上实测，其结果不证明能扩展到大码。

| 条件 | 变体 | 调用数 | 合法帧数范围 | 错帧数范围 | 每次帧数 | runner与初始化之和的平均秒 |
|---|---|---:|---:|---:|---:|---:|
| native32k-a-esm1 | gpu-minsum | 1 | 8 | 0 | 8 | 1.194 |
| native32k-a-esm1 | gpu-spa | 1 | 8 | 0 | 8 | 0.803 |
| native32k-a-esm1 | initial-bp | 1 | 8 | 0 | 8 | 0.805 |
| native32k-a-esm1 | lsd | 1 | 8 | 0 | 8 | 0.854 |
| native32k-a-esm1 | penalty-ramp | 2 | 4..6 | 2..4 | 8 | 3.944 |
| native32k-a-esm1 | warm-fixed | 2 | 8 | 0 | 8 | 3.697 |
| native32k-b-esm1 | gpu-minsum | 1 | 8 | 0 | 8 | 1.239 |
| native32k-b-esm1 | gpu-spa | 1 | 8 | 0 | 8 | 0.796 |
| native32k-b-esm1 | initial-bp | 1 | 8 | 0 | 8 | 0.809 |
| native32k-b-esm1 | lsd | 1 | 8 | 0 | 8 | 0.851 |
| native32k-b-esm1 | penalty-ramp | 2 | 3..4 | 4..5 | 8 | 3.944 |
| native32k-b-esm1 | warm-fixed | 2 | 8 | 0 | 8 | 3.687 |
| quadratic16k-a-esm1 | gpu-minsum | 1 | 8 | 0 | 8 | 0.946 |
| quadratic16k-a-esm1 | gpu-spa | 1 | 8 | 0 | 8 | 0.620 |
| quadratic16k-a-esm1 | initial-bp | 1 | 8 | 0 | 8 | 0.514 |
| quadratic16k-a-esm1 | lsd | 1 | 8 | 0 | 8 | 0.520 |
| quadratic16k-a-esm1 | quad-pair-ramp | 2 | 0 | 8 | 8 | 4.600 |
| quadratic16k-a-esm1 | quad-pair-ramp-warm | 2 | 8 | 0 | 8 | 4.124 |
| quadratic16k-b-esm1 | gpu-minsum | 1 | 8 | 0 | 8 | 0.951 |
| quadratic16k-b-esm1 | gpu-spa | 1 | 8 | 0 | 8 | 0.706 |
| quadratic16k-b-esm1 | initial-bp | 1 | 8 | 0 | 8 | 0.506 |
| quadratic16k-b-esm1 | lsd | 1 | 8 | 0 | 8 | 0.553 |
| quadratic16k-b-esm1 | quad-pair-ramp | 2 | 0 | 8 | 8 | 4.595 |
| quadratic16k-b-esm1 | quad-pair-ramp-warm | 2 | 8 | 0 | 8 | 4.123 |
| wifi648-heldout-esm2 | gpu-minsum | 1 | 34 | 30 | 64 | 0.827 |
| wifi648-heldout-esm2 | gpu-spa | 1 | 42 | 22 | 64 | 0.537 |
| wifi648-heldout-esm2 | initial-bp | 1 | 34 | 30 | 64 | 0.344 |
| wifi648-heldout-esm2 | lsd | 1 | 64 | 21 | 64 | 0.659 |
| wifi648-heldout-esm2 | osd | 1 | 64 | 22 | 64 | 6.806 |
| wifi648-heldout-esm2 | penalty-ramp | 2 | 3..6 | 58..61 | 64 | 2.452 |
| wifi648-heldout-esm2 | warm-fixed | 2 | 36 | 28 | 64 | 2.431 |

接收求解器在$He=\sigma$下最小化$\sum_i q_i e_i$，
其中$q_i=\operatorname{round}(256\min(|LLR_i|,32))$。
它不取得带符号LLR、硬判决向量或发送真值，评价专用数组另行发布。
信道横轴为$E_s/N_0$，不是推断得到的$E_b/N_0$。

大码原生模型使用32766数据p-bit、没有辅助位；完整二次模型使用16380数据位加16380辅助位，
共32760。固定物理位预算下的原始码长不同，不能据此作同码编码因果比较。
全部最好物理态仍等于初态的暖启动调用数为**8/10**。

可行性、似然和译码正确性分别评价；精确ML判决也可能不同于发送字。
只在真正可行态之间比较似然，有限惩罚不证明全局受约束ML等价。
相对LSD，双方均合法的162次配对帧尝试中，随机方法取得更低量化成本的次数为**0**。
重复种子仍是既有噪声上的重复尝试，不能推导大码全局ML或罕见错误率保证。

![WiFi错帧数](figures/rtx5090.png)

[方法](README.zh-CN.md) · [核心数据](../../data/ldpc/rtx5090/calls.json) ·
[核验](../../data/ldpc/rtx5090/verification.json) · [复现](../reproduction.zh-CN.md)

## 参考来源与软件

下列入口保留研究中使用的背景及未实施对照。引用不表示复现；
方法说明中的版本与获取边界继续适用，本次发布没有重新进行一次全文文献审查。

- R1: [Error-Rate Reduction in LDPC Decoding via Bit-Aligned Temporal Reinforcement in Parallel Probabilistic-Bit Dynamics](https://arxiv.org/html/2609.12389v1)
- R2: [https://github.com/nonizawa/pSA_LDPC/blob/99bde4cd173ce90bdde098873925589fb9d4a05d/README.md](https://github.com/nonizawa/pSA_LDPC/blob/99bde4cd173ce90bdde098873925589fb9d4a05d/README.md)
- R3: [quantumgizmos/ldpc](https://github.com/quantumgizmos/ldpc)
- R4: [Localized statistics decoding for quantum low-density parity-check codes](https://arxiv.org/abs/2406.18655)
- R5: [Tavildar LDPC](https://github.com/tavildar/LDPC)
- S1: [Python](https://www.python.org/)
- S2: [NumPy](https://numpy.org/)
- S3: [PyTorch](https://pytorch.org/)
- S4: [Triton](https://github.com/triton-lang/triton)
- S5: [CUDA](https://docs.nvidia.com/cuda/)
- S6: [SciPy](https://scipy.org/)
- S7: [Matplotlib](https://matplotlib.org/)
- S8: [PyYAML](https://pyyaml.org/)
- S9: [uv](https://docs.astral.sh/uv/)
- S10: [GCC](https://gcc.gnu.org/)
- S11: [pytest](https://docs.pytest.org/)
- S12: [Ruff](https://docs.astral.sh/ruff/)
