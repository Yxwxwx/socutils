# BAGEL ZCASSCF 与无 Kramers、无 CD 的 X2C-DMRG-SCF

历史研究记录：文中提到的 Super-CIPT 和轨道 DIIS 实现现已移除。

2026-09-22。目标算例：`/home/Yxwxwx/new-dmrgscf/laser/CaOH`。
研究对象是轨道优化的收敛性，不比较 BAGEL 4C 与本程序 X2C 的绝对总能量。

## 已定位的问题

CaOH 使用 CAS(9e,16 spinors)、6 个等权重 SGFCPX 态、DMRG M=1000、
原始轨道顺序和完整积分。其 404 个 spinor 中有 20 个 inactive spinor，
共有 13,568 个独立复轨道变量。外层能量/梯度阈值为 `1e-8 / 1e-4`，
轨道生成元 Frobenius 范数上限为 `0.2`。

归档日志 `dmrg.before-247990.out` 显示：50 次轨道更新没有达到外层标准；
Macro 49（最后一次更新前）的梯度仍为 `1.928e-3`。这次运行已经开启旧版
adaptive，轨道 Davidson 和 DMRG 均收敛。早期很多步先得到很大的解，再把
所有方向整体缩小到约 7%–14%。因此这次失败不能用增加 Davidson 迭代数解决。
它也不证明不存在收敛解：最后数步的能量和梯度仍在下降。

此前的 TlH 缓存则暴露了另一层问题：普通 Davidson 在原始病态激发度量中
求解失败；这和 CaOH 外层缓慢需要分别验证。原实现还存在复数 RDM 转置、
反向耦合的共轭及预条件对角线不一致问题，详见
`../.tmp_superci_debug/superci_isolated_audit_results.txt`。

## BAGEL 中应借鉴的部分

| 内容 | BAGEL 源码 | 对 X2C-DMRG-SCF 的意义 |
| --- | --- | --- |
| 二阶轨道更新 | `ref/bagel/src/multi/zcasscf/zcassecond_compute.cc:255` 的 `compute_hess_trial` | 包含旋转引起的 Coulomb/exchange 响应和 2-RDM 的 Q′/Q″ 项；新 `second_order()` 包含这些响应，混合单电子 Super-CI 没有完整曲率 |
| 复数切空间 | `ref/bagel/src/util/math/aughess.h:143` 附近的实部投影 | 真正轨道 Hessian 的作用是 `A x + B x*`，应使用实切空间内积或实部/虚部双倍表示；不能直接当作当前复线性的 Super-CI 算符 |
| 求解时控制步长 | `ref/bagel/src/util/math/aughess.h:61` 的 `compute_lambda_` | 在投影 AH 方程中调节尺度，然后构造轨道步；避免先求出过大向量再统一缩放 |
| 内层精度与步长关联 | `zcassecond_compute.cc:150` | `err` 为残差除以 AH 尺度，停止条件为 `err < max(thresh_micro, stepsize*thresh_microstep)`，后者默认系数 `1e-4`；当前仍以未缩放残差 `1e-8` 严格验收 |
| 自然轨道 | `zcassecond_compute.cc:499`、`:529` | BAGEL 同时变换轨道与 RDM。有限 M 的 DMRG 中不能只旋转轨道而保持 MPS 不变；当前仅变换求解坐标更稳妥 |

算法文献：[Reynolds, Yanai, Shiozaki, JCP 149, 014106 (2018)](https://arxiv.org/abs/1804.06470)，
尤其是复数 Hessian 的式 (12) 和带缩放的增广方程。
默认二阶算法及输入说明见 [BAGEL 官方手册](https://nubakery.org/multi/zcasscf.html)。
本表的实现细节以仓库中的本地源代码为依据。

BAGEL 当前默认 `ZCASSecond` 强制 Kramers 对称性；`ZCASSecond_London` 的
`impose_symmetry` 为空。这里借鉴的是共享的二阶优化数学，不引入 Kramers
投影、正负能轨道旋转、London 磁场或 DF/CD 依赖。X2C 已有固定的两分量
Hamiltonian，轨道变化应对这一 Hamiltonian 求导。

## 本次先修复的共同基础

1. **将普通求解与 adaptive 步长策略解耦。** 在完整积分、无 Kramers、
   无冻结/筛选且 `canonicalize_=False` 的路径，两者共享经独立行列式投影
   验证的 Hermitian 混合单电子 Super-CI 算符、正确复数梯度以及完整占据
   度量坐标。`superci_adaptive=False` 现在也能用这套数值预处理，仍求无移位
   方程。其他积分/旋转路径保留原有实现。
2. **adaptive 以实际轨道步长判断是否移位。** 删除“度量条件数低于
   `1e8` 就直接采用无移位大步”的分支。原来所谓健康度量的 CaOH 正好被
   该分支放过。现在只有解已满足半径时才使用零附加移位。
3. **修正新路径的外层一阶能量预测。** 本程序 `g` 满足
   `dE = 2 Re(g† x)`，旧 `0.5 Re(g† x)` 使一阶预测小了 4 倍。
   Super-CI 分子矩阵不是轨道 Hessian，因此没有把它代入二阶 Newton
   预测公式。其他既有路径保留原步长启发式，避免扩大此次修改范围。

占据度量正交化没有更改活性轨道或 MPS。无移位和移位求解都检查原坐标残差
及保留坐标残差。移位残差明确属于 `(H + μ I)` 方程，不冒充无移位收敛。
外层能量、梯度、DMRG 和 Davidson 的验收阈值没有放宽。

### DMRG 重启的独立故障

CaOH 对照 `248004` 的前 30 次更新均降低能量；切入两点 MPS 的 warm
restart 后，Block2 报告的根能量与拆分 MPS 的 Hamiltonian 期望值出现
`7.05e-4 Eh` 不一致。旧实现只警告，仍将 CI 标记为收敛，导致下一次外层
能量上升 `1.19e-4 Eh`。再下一步因单根能量变化未达标而终止。
原目录中的 `247999` 和本次二阶 `248010/248012` 也遇到了重启问题。

已在共同的 `DMRGCI.kernel` 修复：不一致的两点结果必须标记为未收敛；
内部 warm restart 未通过验收时，释放旧 Block2 frame，以原有完整 cold
schedule 重算一次。cold 仍失败则保持失败，不能循环重试或放宽阈值。
不改变 M、态权重、两点 endpoint、轨道排序或 DMRG 精度设置。
`convergence_info['restart_fallback']` 记录失败原因。
回归测试在真实 Block2 计算后注入错误报告能量，检查成功回退与冷启动
仍失败两种情况，确保错误能量不会被当作收敛结果。

混合单电子算符仍是 Super-CI 近似。它与下述真正二阶路径分开，不能把
`superci_adaptive=True` 称为 BAGEL 二阶算法。

## 新增真正的二阶路径

入口为 `mc.second_order()`，实现位于 `mcscf/zmc_ah.py`，复用原有
MCSCF 外层、积分容器和用户配置的 DMRG 求解器，不需要 3/4-RDM。
它对固定 X2C Hamiltonian 和固定 1/2-RDM 的能量求轨道导数：

* 密度响应采用已有完整积分 AO `get_jk`，包括 inactive 和总密度的响应。
* 完整 MO 积分块 `(pp|aa)`、`(pa|pa)`、`(pa|ap)` 补齐 Q 的各指标导数。
  `(aa|pp)` 先变换活性指标，再利用电子对交换得到 `(pp|aa)`，避免
  先变换两个全空间指标造成巨大中间文件。三块占用约
  `48*nmo**2*ncas**2` 字节，CaOH 约 2 GB；超出内存预算会明确报错。
* 梯度移动坐标的导数加入 `0.5 [X,G]` 后成为当前指数坐标中的 Hessian。
  返回算符是能量实坐标 Hessian 的一半，与现有半梯度 `g` 配套，
  因此二阶预测为 `2 Re(g†x) + Re(x†H x)`。
* 与 BAGEL `AugHess` 一样，Krylov 投影使用 `Re(u†v)`，求解
  `[[0,gᵀ],[g,H/λ]]`，再用 `x=v_orb/(λ*v_ref)` 构造轨道步。
  在小投影矩阵中调节 `λ≥1`，不为每个 λ 重做昂贵 Hessian 作用。

保持无 Kramers、无 CD/DF、无冻结/筛选轨道、不做活性自然轨道旋转。
使用已有 `max_stepsize=0.2` 的生成元 Frobenius 范数约定，而非直接照搬
BAGEL packed-vector 半径 `1.0`。这也不是 BAGEL 的 4C 积分或负能轨道实现。

独立检查包括真实复 spinor 的固定 RDM 能量一阶/二阶中心差分、完整
Hessian-vector 梯度差分、实切空间对称性、`H(ix)≠iH(x)`、core/virtual
复相位协变性，以及正定和不定的显式双倍实矩阵对 AH 解和步长的核对。
公开 `mc.second_order()` 入口和二阶能量预测也纳入检查。
79 项相关回归通过，4 项未选择，耗时 19.80 秒。命令如下：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_dmrgci.py tests/test_zmc_second.py \
  tests/test_superci_adaptive.py tests/test_superci_cholesky.py \
  tests/test_zmcscf_canonicalization.py tests/test_supercipt.py \
  -k 'not cholesky_variants'
```

## 数值验证

### 固定起点的一步 CaOH 实验

脚本：`../.tmp_superci_debug/caoh_bagel_probe.py`；Slurm 作业 `248003`。
输入为归档 `mcscf.chk:mo_coeff_iter_1`，新做同设置的 DMRG，两个候选使用
完全相同的轨道、RDM、梯度和积分，半径均为 `0.2`。这不是从 HF 开始的
完整轨道优化，也不涉及 BAGEL 4C 的能量。

| 指标 | 原 adaptive 的健康度量分支 | 方程内限制轨道半径 |
| --- | ---: | ---: |
| 初始梯度 | 0.05070560 | 0.05070560 |
| 后处理缩放系数 | 0.1107528 | 1.0 |
| 实际生成元范数 | 0.2000000 | 0.1994006 |
| 一步 ΔE / Eh | −0.000378974 | −0.002152507 |
| 更新后的梯度 | 0.04684920 | 0.01949742 |
| 所有移位尝试的 Davidson 迭代数 | 9 | 171 |

后者单步下降约为前者的 5.68 倍，但内层工作更多，不能据此声称总时间更短。
旧步方向的固定 RDM 中心差分给出 `−0.000388135334`，正确一阶预测为
`−0.000388135063`，差 `2.71e-10`；旧预测为 `−0.000097033766`。
完整记录位于 `../.tmp_superci_debug/caoh_bagel_probe_248003/result.json`。

### 普通 Davidson 的失败缓存回归

使用先前 exact-CASCI TlH 缓存中的 F/L、密度和 2-RDM，重建修正后的共同
Super-CI 算符；没有重跑整个 TlH 分子计算，也没有将本次 CaOH 的 DMRG
替换为 FCI。无附加移位的 Davidson 4 次迭代收敛：原坐标残差
`2.80e-12`，占据坐标残差 `1.52e-9`，阈值 `1e-8`，保留全部 46,188 个
变量。原算例的旧实现曾在 500 次迭代后失败。
这同时改变了算符修正和预条件，不能把全部改善归因于单独一个因素。
无移位解的范数仍很大（约 3018），所以这里只证明内层求解通过。
记录：`../.tmp_superci_debug/tlh_unshifted_repaired.json`。

### 完整 CaOH 配对计算

`248004`：新 bounded 策略，最多 80 次更新；`248005`：普通无移位路径，
前 12 次更新的对照。两者从相同归档轨道重启，均为 8 核、同一 DMRG
schedule、固定随机种子、相同验收阈值。输入轨道散列及逐轮记录保存在各自
`../.tmp_superci_debug/caoh_bagel_probe_<job>/result.json`。
`248004` 在 DMRG 重启故障处停止；`248005` 按计划完成 12 次更新，所有
普通 Davidson 均收敛，但尚未达到外层标准。普通路径仍需整体缩放大步，
不将这 12 步的内层成功称为外层收敛。

二阶计算 `248010` 从同一早期轨道出发，首步梯度降至 `1.673e-3`，
实际/预测能量变化比为 `1.008`，AH 残差 `7.73e-9`，耗时约 232 秒。
它也遇到旧版 DMRG 重启故障，在第 4 次更新后停止，前 3 次更新可靠。
`248012` 从归档第 50 步轨道出发，前两次更新将梯度从 `1.095e-3` 降至
`2.279e-4`；第三次的 DMRG restart 失败。`248007` 在发现首个全空间
积分变换产生过大中间文件后主动停止，改成电子对交换顺序再提交。

修复 DMRG 验收后的续算：`248020` 从 `248004:mo_coeff_iter_30` 的可靠
轨道继续 bounded；`248021` 从 `248012:mo_coeff_iter_2` 的可靠轨道继续
二阶。续算重新做初始 DMRG，Super-CI 梯度倍率也从默认 0.5 开始，因此
分段迭代数不能冒充同一起点、连续运行的严格加速比。

**bounded 已完成收敛验证。** `248020` 再做 7 次更新后，最终
`|ΔE|=6.919e-9 Eh`、`||g||=2.814e-5`，两项均满足原来的
`1e-8 / 1e-4` 标准，所有轨道 Davidson 通过严格残差验收。
最后一次更新遇到 DMRG warm restart 未收敛，实际执行了冷启动回退并
重新验收。该轨迹由早期 30 次有效更新与修复后 7 次续算组成；原 adaptive
归档在 50 次更新时仍未达到外层阈值。末步有 `6.919e-9 Eh` 的微小能量
上升，故不宣称整个轨迹严格单调下降。

**二阶停滞点续算也已收敛。** `248021` 再做 3 次更新后，最终
`|ΔE|=5.997e-9 Eh`、`||g||=1.977e-5`。从原归档第 50 步轨道开始，
有效轨迹为 `248012` 的 2 次更新加 `248021` 的 3 次更新，共 5 次。
所有 AH 未缩放残差均小于 `1e-8`；修复后的续算实际执行了两次失败
warm restart 的 cold 回退，其中一次明确发现 `H-SE=1.571e-5 Eh` 的
能量/MPS 不一致并拒收。每个宏迭代约 268–273 秒，不能声称比 Super-CI
更省总时间。二阶从早期轨道开始的完整连续收敛尚未验证。

| 已完成路径 | 最终能量变化绝对值 / Eh | 最终轨道梯度 | 轨迹说明 |
| --- | ---: | ---: | --- |
| 修复后的 adaptive Super-CI | `6.919e-9` | `2.814e-5` | 相同早期轨道，30 次有效更新 + 7 次续算 |
| BAGEL 思路的二阶 AH | `5.997e-9` | `1.977e-5` | 原第 50 步轨道，2 次有效更新 + 3 次续算 |

最终记录：`../.tmp_superci_debug/caoh_bagel_probe_248020/result.json`、
`../.tmp_superci_debug/caoh_bagel_probe_248021/result.json`。
建议现有 CaOH 驱动先保留 `mc.superci_adaptive=True; mc.superci()`，它会
直接使用修复后的共同实现。`mc.second_order()` 保留为二阶比较入口，
不默认替换为开销更高且尚未完成早期起点全程比较的算法。
本次没有运行 BAGEL 的 4C 数值算例，BAGEL 对照指源代码与算法对照。

## 当前边界

完整积分 Hessian 每次宏迭代需额外 AO→MO 变换和每次微迭代的 JK 响应，
不能仅凭更少的宏迭代宣称总耗时更小。验收针对固定 RDM 曲率；重新优化
CI 后的梯度差分是 relaxed Hessian，不应混作同一个测试。

外层尚无实际能量上升时的事务式回退。Super-CI 会接受上升再调小梯度
倍率；二阶路径当前接受有界 AH 步。真正回退必须同步处理轨道、能量、
CI/MPS、RDM 缓存和重启调度。这不是本次已经完成的功能。
