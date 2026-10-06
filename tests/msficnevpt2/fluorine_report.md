# F 原子 one-step X2C-MS-FIC-NEVPT2：六态与 4+2 分组

2026-10-06，两种计算均完成，退出码为 0。本报告记录原始计算结果，
不通过平均 RDM、强行平均能量或事后清零劈裂恢复简并。

## 设置与比较范围

- F，dyallv3z，CAS(7e,16 spinors)，六态等权 SA-DMRG-SCF，M0=1000。
- 从已完成的 `p-splittings/17/F/dmrg_state` 恢复轨道及实际 MPS，
  不重新运行 HF/DMRG，不以 FCI 参考替换 MPS。
- 完整 ERI，无 CD/DF，无 Kramers restriction；沿用保存参考的 Fiedler 顺序。
- 两种计算共用同一六态 SA-Fock/Dyall。联合计算的 model roots 为 0–5；
  分组计算分别取 0–3（P3/2）及 4–5（P1/2），求解前同时限制 IC
  generators 和 target roots，不是对联合计算的 Heff 作事后切片。
- η 是实能级位移（Eh），不是 IPEA。非零 η 使用完整矩阵 norm correction。
- 两种计算均使用 `metric_refinement=False`，保持本轮比较一致。
  当前 production 默认的 Gram 精修尚未做完整 F 能量比较，不能将本表
  作为新精修算法的完整分子验收。

参考 Hamiltonian 指纹：
`d541f49b5e91b37fea18a571bed3105f1d520c02adfb0c3092fbb87369ca957c`。

共同 preparation 指纹：
`5d25a39af72b29dab5802e3200919d79117f3f6c81b9e7495586b95f262a1c11`。

## 无位移 MS-MR 的六个原始能量

单位 Eh，各 PT 列是对应 Heff 的本征值，不表示逐行跟踪原始 MPS root。

| State | MCSCF | 六态联合 MS-MR，η=0 | 4+2 MS-MR，η=0 |
|---|---:|---:|---:|
| 0 | -99.57346809992214 | -99.73644798377252 | -99.73644715519927 |
| 1 | -99.57346809992026 | -99.73644798376475 | -99.73644715519295 |
| 2 | -99.57346809991938 | -99.73644798376188 | -99.73644715519077 |
| 3 | -99.57346809991785 | -99.73644798375760 | -99.73644715518682 |
| 4 | -99.57162922272208 | -99.73461638223637 | -99.73461317904835 |
| 5 | -99.57162922272101 | -99.73461638221752 | -99.73461317903596 |

## 劈裂与精细结构间隔

单位 cm⁻¹，使用 219474.63137 cm⁻¹/Eh。组内劈裂是最大值减最小值；
两组间隔按各多重态的能量中心报告，这只是统计量，未修改原始能量。

| ansatz | model space | η / Eh | P3/2 组内劈裂 | P1/2 组内劈裂 | P1/2 − P3/2 间隔 |
|---|---|---:|---:|---:|---:|
| SS-SR | 六态联合 | 0.2 | 1.03825040 | 0.00624751 | 402.182246 |
| SS-SR | 4+2 分组 | 0.2 | 1.03884425 | 2.85101e-5 | 402.168942 |
| MS-MR | 六态联合 | 0.2 | 3.27487e-6 | 4.11698e-6 | 402.004592 |
| MS-MR | 4+2 分组 | 0.2 | 2.73529e-6 | 2.71658e-6 | 402.524101 |
| SS-SR | 六态联合 | 0.0 | 1.02943664 | 0.00617368 | 402.168707 |
| SS-SR | 4+2 分组 | 0.0 | 1.03002006 | 2.85724e-5 | 402.155528 |
| MS-MR | 六态联合 | 0.0 | 3.27487e-6 | 4.13569e-6 | 401.990072 |
| MS-MR | 4+2 分组 | 0.0 | 2.73218e-6 | 2.71970e-6 | 402.511240 |

MCSCF 间隔为 **403.586895 cm⁻¹**。4+2、η=0 的 MS-MR 将其降低
**1.075656 cm⁻¹（0.2665%）**；相比六态联合，分组间隔增加
**0.521167 cm⁻¹**。这些是内部方法比较，未作为实验精度验证。

## 数值边界与复现

参考 MPS 的最大实际 CAS 残差为 7.37e-9 Eh。计算中仍有 retained IC F
厄米性及 IC 相对残差 warning，均保留诊断，没有忽略 NaN、负 metric、
discarded-space source 或近奇异分母等硬错误。组内劈裂属于当前数值精度，
不应宣称解析到了更小的真实物理效应。

[fluorine.py](fluorine.py) 使用已有 F checkpoint；准备完成后两种 ansatz、
两种位移及后续分组均复用相同 `prepared.pkl`，不重新计算 transition RDM。
从仓库根目录执行：

```bash
ulimit -s unlimited
ulimit -c 0
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
.venv/bin/python -u -m tests.msficnevpt2.fluorine
.venv/bin/python -u -m tests.msficnevpt2.fluorine --grouped
```

原始结果是本地 `fluorine_run/results.json` 和 `grouped_results.json`，
含完整 Heff、class corrections、norm、J² 与数值诊断。缓存、日志、RDM
及 checkpoint 留在被 Git 忽略的运行目录，不进入源码提交。
本报告保留关键结果用于版本化；原始数据和续算缓存并未删除。
