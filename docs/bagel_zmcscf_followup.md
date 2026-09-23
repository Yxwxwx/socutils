# BAGEL ZCASSCF 再次对照：改进优先级

历史研究记录：文中提到的 Super-CIPT 和轨道 DIIS 实现现已移除。

2026-09-22。范围为无 Kramers、无 CD/DF 的 X2C-DMRG-SCF。
下文先保留修改前的源码审查与性能证据；文末记录本轮实现及验证结果。
审查部分的源码行号对应修改前版本。

## 当前证据

F 的新验证目录：
`/home/Yxwxwx/new-dmrgscf/p-splittings/17/F/optimizer_validation_20260922`。
两组从同一份新 HF 轨道开始，CAS(7e,16 spinors)、六态等权、M=1000。
外层阈值保持 `|dE|<1e-8`、独立轨道变量梯度范数 `<1e-4`。

| 路径 | 轨道更新 | 优化计时 / s | 轨道内层总迭代 | 接近 0.2 步长上限的更新 | 对原 Super-CI 六态最大差 / Eh |
|---|---:|---:|---:|---:|---:|
| adaptive | 25 | 803.50 | 1670（包括全部移位尝试） | 17 | 3.834e-9 |
| second_order | 23 | 961.53 | 408 | 17 | 5.147e-10 |

计时来自各自 `result.json` 的 `wall_seconds`；包含首次 CASCI，排除共享 HF。
两种内层算符开销不同，1670 和 408 不能直接换算为耗时。
二阶前 17 个更新的实际/预测能量变化比在 1.008–1.086，均降低能量。
这支持测试更大的步长，但不能证明更大步长一定安全或更快。

## P0：先修复重启调度与宏迭代回调的接口

`dmrg/dmrgci.py:759` 以 `"accepted" in environment` 判断是否启用结构化保护。
但是 `mcscf/zmc_superci.py:1303` 创建的记录及 `:1676` 传出的回调没有这个字段。
F 的 adaptive 26 条、second_order 24 条记录均缺少它。
因此这两条实际路径仍走仅由梯度/密度触发的旧调度，未启用大步长、终止记录等保护。
已有 `tests/test_dmrgci.py:166` 使用手工构造且带 `accepted` 的字典，未覆盖真实接口。

最小调用验证：设 `dmrg_switch_tol=1e-3`，输入梯度 `1e-4`、步长 `0.2`、
`ci_solver_converged=True`。不带 `accepted` 时返回重启 True；加上
`accepted=True` 后返回 False，并记录 `orbital_step_too_large`。

修复应统一真实回调记录与调度器，并核对调用时机：当前回调在新 CASCI/RDM
计算之后才执行，它控制下一次 kernel，所用的步长却是刚刚执行过的那一步。
不能仅加一个字段就宣称下一次试探步已经受保护。应增加使用实际宏迭代回调的
集成检查，覆盖小梯度大步、终止记录以及 warm 失败后的 cold 回退。
既有能量/MPS 一致性验收与 cold 回退必须保留。

## P1：步长控制需要真实的接受/拒绝和可调整半径

BAGEL `ref/bagel/src/util/math/aughess.h:58` 使用 packed-vector 半径 1.0，
在投影 AH 中调节 lambda。当时的二阶 AH 使用
`max_stepsize/sqrt(2)`，当前输入固定为 0.2 的生成元 Frobenius 范数。
两者的数字和 4C/Kramers 变量空间不能直接互换。

我们虽然计算了二阶实际/预测下降比，但 `mcscf/zmc_superci.py:1615`
对 second_order 总是执行 `accepted / scaled AH`；`:1590` 的拒绝分支禁用。
日志里的 `Trust=0.5` 在二阶路径不控制 AH 半径，容易误导诊断。
BAGEL 这条 `ZCASSecond_base::compute` 本身也没有基于新能量的回滚循环；
接受/拒绝是适合我们 DMRG 路径的进一步改进，不能说是它已有的功能。

建议先实现同步处理 MO、能量、RDM、MPS 和重启状态的接受/拒绝，再测试
0.2 起步、可靠下降时逐渐扩至 0.3/0.4、模型不可靠时收缩的半径。
拒绝后从已接受的点重算，不能仅恢复 MO 而继续使用试探点 MPS。
现有 `mcscf/orbital_linesearch.py` 和 `zmc_supercipt.py` 已有试探点及恢复点
的处理模式，可复用；无需再造通用优化框架，也不必默认多做整套 Wolfe 搜索。

## P1：按轨道步长调节内层精度

BAGEL `zcassecond_compute.cc:141–150` 的判据为

```
norm(residual) / lambda < max(thresh_micro, stepsize * thresh_microstep)
```

其中 `thresh_microstep=1e-4`（`zcassecond.cc:38`）。我们的
旧版 AH 每轮都要求未缩放残差 `<1e-8`，远离收敛时可能过度求解。

对 F 的已完成日志逐轮统计：采用类似条件
`residual/lambda <= max(1e-8, packed_step*1e-4)` 时，第一次越过阈值的位置
合计为 193 次，而实际严格求解用了 408 次。这里只沿旧轨迹读取日志：
提前停止会改变步方向和后续轨迹，不能据此声称实际迭代减半或耗时减半。
BAGEL 与我们的梯度归一化也需明确换算。

建议采用早期适度精度、接近外层收敛时恢复严格精度的策略；记录实际残差、
有效阈值和停止原因，保持外层能量/梯度及 DMRG 最终验收标准。
不能把非精确的早期求解继续标成“残差达到固定 1e-8”。

## P1：减少每次 Hessian 作用及积分变换的重复开销

BAGEL 在 `zcassecond_compute.cc:88–105` 准备半变换积分并传给微迭代，
`aughess.h:124–135` 增量更新投影矩阵。其 DF 数据结构不能直接用于本任务，
可以借鉴缓存和增量计算的方式。

本程序可先做不改变数学目标的改动：

1. 旧版 AH 的 inactive/total 两个响应密度合并成一次批量 JK。
   现有 `spinor_hf.SCF.get_jk` 支持该批量格式；H2/STO-3G 两个随机复 Hermitian
   密度的 full-ERI 检查中，批量与分开调用的 J/K 最大差为 0。
   这验证接口和结果一致性，尚未测得实际速度收益。
2. 求得 AH 系数时已持有 `sigma @ coeff = Hx`。将该结果或二次型用于能量预测，
   避免 `zmc_superci.py:1560` 再调用一次 `hop(applied_x)`。
   只有最终应用方向与求解方向一致时才能复用；若缩放/变换方向，必须同步修正。
3. 旧版 AH 每次复制全部 Krylov 向量并重算整个投影矩阵；
   可像 BAGEL 一样仅补新行列，并保留实内积及对称化。
4. `zmc_ao2mo.py:897` 虽计算了可用内存，但后续 `nrr_outcore.general` 未传入；
   旧版 AH 也未传入。当前环境中即使 `PYSCF_MAX_MEMORY=500000`，
   此函数的默认 `max_memory` 仍为 4000 MB。应传入扣除驻留数组和 DMRG 占用后
   的有效预算，不能把 500 GB 同时完整分给每个模块。
5. `papa` 和 `paap` 的第一对 MO 相同，可评估共用半变换；`aapa` 和 `aapp`
   同样共用 `(a,a)` 第一对。现有 `nrr_outcore.general` 每次都会单独调用
   `half_e1`。跨不同宏迭代不能直接复用旧 MO 积分。

## P2：改进 adaptive 移位搜索和二阶预条件

`zmc_superci_adaptive.py:41–46` 每次改变 shift 都重启完整 Davidson，先扫描
数量级，再二分；F 因此合计 1670 次迭代。可借鉴 BAGEL 在小投影问题中
调整参数的方式，复用已有 Krylov 子空间，或先复用上次移位和初始方向。
这里白化坐标中的移位是 `mu * B†B`（本实现为 `shift_diag`），不是 `mu*I`；
投影及最终原坐标/保留坐标残差都必须对应真正的移位算符。

BAGEL `zcassecond_compute.cc:62,499,529` 在自然轨道基中做微迭代并变换 RDM；
`:457` 的预条件分母仍标为近似对角线。我们的二阶近似对角线来自当前基的
密度/Fock/Lagrangian 对角元，尚未利用自然轨道坐标改善预条件。
可以仅在内层预条件的坐标中对角化活性密度，并可评估 core/virtual 的块预条件；
保持物理 MO 和有限 M 的 MPS 不变，再把方向映回原坐标。
不能直接打开 `mc.natorb=True` 并保持旧 MPS。

## 对照范围与验证顺序

BAGEL 这条二阶路径在每个宏迭代先求 CI/RDM，随后固定 RDM 求轨道 AH；
它没有在微迭代中加入 DMRG/CI 响应。联合轨道–CI Newton 是更大的后续项目。
它还以梯度 RMS 收敛（`rotfile.h:92` 附近定义为 norm/sqrt(size)），本程序
使用独立变量梯度范数并同时验收能量变化，阈值数字不能直接照搬。

建议次序：修回调接口 → 做等价的重复计算消除 → 分别测试内层自适应精度、
带接受/拒绝的半径调整 → 再评估预条件和子空间复用。
每次只改变一项，先使用 F 六态回归，再用同一起点的 CaOH、UNF3 比较。
指标同时包含六态能量、外层梯度/能量变化、DMRG 一致性、宏迭代、
Hessian 作用次数、失败回退次数和总耗时，不能只比较宏迭代数。

## 本轮实现

上述 P0–P2 已应用到无 Kramers、无 CD/DF 路径：

- 在实际试探 CASCI **之前**，用即将执行的步长、当前梯度和 CI 状态设置 MPS
  重启条件。拒绝后的重试强制冷启动；终止记录不能触发下一次 warm solve。
- adaptive 与 second_order 从 `orbital_trust_start=0.2` 起步，根据实际/预测
  下降调节半径，受 `max_stepsize` 硬上限约束。试探失败时缩小半径；六次都失败
  或后续出错时，在已接受轨道上冷启动重算 CI/RDM/checkpoint，保持状态一致。
  这是对本程序 DMRG 路径的补充，不是 BAGEL 原有的能量回滚功能。
- 二阶早期采用 BAGEL 形式的缩放残差判据，`second_order_micro_step_tol=1e-4`；
  梯度不超过外层阈值的十倍时，恢复原始残差 `<=1e-8`。设该参数为零可全程严格求解。
- 批量计算两个 JK 响应，复用已计算的 Hessian 二次型，增量更新投影矩阵；
  `(a,a)` 和 `(p,a)` 半变换分别共用。变换显式传入扣除驻留量后的内存预算。
- 二阶预条件使用自然轨道/半正则坐标，物理 MO、RDM 和 MPS 不旋转。
  adaptive 的所有移位在同一 Krylov 子空间中求解，使用实际的 `B†B` 移位投影，
  并同时检查原坐标和保留坐标残差。

默认硬上限仍为 0.2；F 新测试显式设为 0.4，以验证半径增长。这里没有引入
轨道–CI 联合响应，也没有采用 BAGEL 的 4C、Kramers 或 DF 数据结构。

## 实现检查

相关回归共 **85 passed, 4 deselected**。四项未选中的是本轮范围外的 Cholesky
参数组合。新增验证包括：

- 随机复轨道的四个 MO 积分块与独立 `nrr_outcore.general` 结果一致，
  半变换调用从四次降到两次；批量 JK 及固定 RDM Hessian 有限差分保持一致。
- 复数、病态度量的小型独立矩阵检查 adaptive 的移位、步长和两类残差；
  检查二阶严格/非精确停止判据、预条件相位协变性及缓存二次型。
- 三个真实 Block2 小分子试验强制拒绝轨道步，覆盖重试和六次耗尽后的恢复；
  最终 MO、live MPS/RDM 能量及 checkpoint Hamiltonian 相互一致。

运行命令（先加载 anaconda3/openmpi 模块）：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_dmrgci.py tests/test_zmc_second.py tests/test_superci_adaptive.py \
  tests/test_superci_transactions.py tests/test_superci_cholesky.py \
  tests/test_zmcscf_canonicalization.py tests/test_supercipt.py -k 'not cholesky_variants'
```

## F 原子最终回归

目录：`/home/Yxwxwx/new-dmrgscf/p-splittings/17/F/optimizer_bagel_improved_20260922`。
Slurm **248054 / 248055** 均为 `COMPLETED, ExitCode=0:0`，cu03、16 核、500 GB。
新旧两轮使用相同 HF checkpoint/MO 哈希、CAS(7e,16 spinors)、六态等权、
M=1000、DMRG schedule、随机种子及外层阈值。每组从新 MPS 开始；不读取之前的
DMRG-SCF checkpoint。输入差异仅为本轮新增策略参数和硬步长上限 0.2 → 0.4。

| 方法 | 轨道更新：旧 → 新 | 宏迭代点数：新 | 计时 / s：旧 → 新 | 内层算符作用：旧 → 新 | 六态对原 Super-CI 最大差 / Eh |
|---|---:|---:|---:|---:|---:|
| adaptive | 25 → **17** | 18 | 803.50 → **530.57** | 1670 → **100** | **1.440e-9** |
| second_order | 23 → **16** | 17 | 961.53 → **526.00** | 408 → **67** | **2.766e-9** |

计时包含首次 CASCI，不包含共享 HF。两轮节点相同；单次运行分别减少约 34.0%
和 45.3%，不是重复测量的统计加速比，也未单独分离各项改动的贡献。
旧二阶的 408 次仅计内层，另有 23 次用于外层能量预测的 Hessian 作用；新二阶
直接复用内层二次型。新 adaptive 的 528 次小矩阵移位求解不需额外大算符作用。

验收逐项通过：

- adaptive：最终梯度 `2.9196e-5`，末次 `dE=-8.2252e-9 Eh`；
  second_order：`1.8553e-5`、`-2.0917e-9 Eh`。
  均满足原有 `|dE|<1e-8` 且梯度 `<1e-4`。
- adaptive 所有内层均满足原/保留坐标严格残差要求，末次 `6.405e-10`。
  二阶前 13 次使用非精确判据，最后 3 次恢复严格原始残差，末次 `1.275e-9`。
- 所有 CI 收敛，所有步长符合当轮半径；试探前重启门控使用实际步长。
  两组各有 4 次允许 warm 初始化，没有 warm 失败回退。
  F 本身未触发拒绝；拒绝/恢复由上述真实 Block2 故障注入检查覆盖。
- 两组运行记录的生产源码 SHA256 相同，并与验收时源码一致；没有 Kramers
  限制、CD/DF 或物理活性轨道自然化。

六态按能量排序，单位 Eh；参考为用户原始 `F/dmrg.out` 对应的
`F/dmrg_state/root_energies.npy`，未修改参考文件。

| 态 | 原 Super-CI | 新 adaptive | 新 second_order |
|---:|---:|---:|---:|
| 0 | -99.573468099922 | -99.573468098490 | -99.573468099974 |
| 1 | -99.573468099920 | -99.573468098485 | -99.573468099972 |
| 2 | -99.573468099919 | -99.573468098480 | -99.573468099972 |
| 3 | -99.573468099918 | -99.573468098478 | -99.573468099970 |
| 4 | -99.571629222722 | -99.571629223128 | -99.571629225488 |
| 5 | -99.571629222721 | -99.571629223123 | -99.571629225487 |

六态都在 `1e-8 Eh` 内吻合，四重态组及二重态组内部最大分裂不超过
`1.24e-11 Eh`。可用测试目录中的 `compare_results.py` 重跑完整验收，结构化结果
保存在 `comparison.json`，逐态差保存在 `comparison.md`。

这完成了 F 回归，尚不能据此断言新版在 CaOH、UNF3 上的收敛轮数。
下一步应使用相同初始 HF、相同 DMRG 精度做分子对照。

## 第三次审查：新版之后还应改进什么

2026-09-22。以下记录**第三次审查时的建议与证据**。审查当时只增加此记录和
独立探针，生产源码哈希仍与 248054/248055 一致；随后按用户要求实施的
第 1、3 项见文末。其他建议仍未实施。

### 1. 优先补齐内层失败后的恢复

当前 `mcscf/zmc_superci.py:1142–1146` 在内层返回 `converged=False` 时立即抛出
异常。六次缩半径重试只覆盖已经得到方向后的试探 CASCI/能量拒绝。因此
`maximum_space` 或 `linear_dependence` 仍会直接终止整个优化。

独立数值探针：固定随机种子 128，24 维实对称正定 Hessian（用 12 个复变量表示），
限制子空间为 4，保持 `tol=1e-8, micro_step_tol=1e-4`：

| 半径 | 缩放残差 | 有效阈值 | 内层成功 |
|---:|---:|---:|---|
| 0.4 | 1.516e-1 | 2.822e-5 | 否 |
| 0.05 | 2.453e-5 | 3.533e-6 | 否 |
| 0.025 | 9.127e-7 | 1.762e-6 | 是 |

直接调用现有 `_bounded_orbital_update`，实际尝试半径只有 `[0.4]`，随后报错。
这证明存在可恢复的失败路径，不代表真实分子必然在 4 维子空间内收敛。

最小改进：对明确的数值未收敛，先收缩半径重试；此时尚未进行新 CASCI，
不需要重算/恢复 MPS。积分损坏、无效输入等结构性错误仍应立即报错。
同一宏迭代内 Hessian/RDM 不变，能量拒绝后的半径重试也可继续使用已有
Krylov 向量和 sigma，当前每次调用 `solve` 都从头开始。

BAGEL 对照：`zcassecond_compute.cc:153–158` 多次重新正交化；
通用 `util/math/davidson.h:132–199` 有投影度量正交化及子空间回收。
**后者是通用 CI Davidson，不是 `AugHess` 自带的重启功能**。
我们的 AH 后续可增加保留 Ritz 向量的有限子空间重启，但应先完成简单的
缩半径恢复，并保持残差验收，不能直接接受未收敛方向。

### 2. CI 验收：补单态一致性，区分阈值与真实残差

BAGEL `ci/zfci/zharrison.cc:255–269` 由 `Hc-Ec` 的实际残差 RMS 判断每个态。
我们的 `dmrg/dmrgci.py:2042–2062` 主要使用逐态扫描能量变化和最终零噪声；
`:2102` 的 `local_residual_bound=sqrt(final_thrd)` 是配置阈值的平方根，
并非实测全局 CI 残差，也不能直接当作轨道梯度误差界。

此外，`:1530–1591` 的 MPS expectation/报告能量一致性检查仅在 `nroots>1`
时运行。UNF3 当前是单态，没有同样的验收覆盖。这是明确的检查缺口，
不是已经发现 UNF3 能量错误。

建议先把单态 `⟨MPS|H|MPS⟩` 与返回能量的比较补到共用验收流程。
多态现有 `H_sub-S_sub E` 只检验选定根空间中的残差，不能替代完整 `Hψ-Eψ`。
最终轨道收敛点再做固定轨道的 CI 加严复核，比较能量、RDM 与重算梯度；
必要时增加 M 收敛或方差诊断。有限 M 的全局方差不必为零，不应照搬 FCI
残差阈值，否则可能把已经在有限 M 流形上优化好的 MPS 错判为失败。

### 3. 仍有确定的重复计算可直接消除

BAGEL `zcassecond_compute.cc:67` 从已有 Jop 取 `core_fock()`。
我们在 `zmcscf.py:45–55` 的 CASCI 准备阶段计算 core JK，随后
`zmc_supercipt.py:274–281` 的梯度构造对同一密度再算一次。
H2/6-31g 的独立调用计数检查确认：两次 core JK 输入完全相同，最大差为 0。
可将 core JK 存在该轨道点的 ERIS 对象上，CASCI 与梯度共用；MO 改变即换对象。

`zmc_superci.py:1745` 还再次执行 `expmat(dr)`，其结果只进入未使用的 `nvar`
及注释代码；真正的轨道更新已在 `:1152` 完成。可直接删去这次重复指数运算。
这两项不改变优化数学或 DMRG 精度，宜先做；实际耗时收益仍需测量。

### 4. 下一轮性能重点应转向 DMRG 扫描

对新版 F 输出按每次 DMRG 的累计 `Time elapsed` 末值求和：

| 方法 | DMRG 扫描累计 / s | 总计时 / s | 扫描占比 | cold / warm 次数 |
|---|---:|---:|---:|---:|
| adaptive | 421.936 | 530.566 | 79.5% | 14 / 4 |
| second_order | 392.090 | 526.003 | 74.5% | 13 / 4 |

此统计只算扫描日志计时，不包含 MPO、RDM、拆分根和 checkpoint 等开销；
因此不能把剩余时间全部归因于 AH。F 每次 cold 用 23 sweeps，warm 固定 8 sweeps。
`:1481` 显式给重启设置 `tol=0`，目的是避免聚合能量先收敛而某个根未收敛，
不能简单删除这个保护。

最小试验可以先使用已有 `restart_sweeps=4`，保持逐态及 expectation 验收和
失败回退，检查六态与最终梯度；无需先建立动态调度框架。
进一步可将“复用已接受的 MPS 作初猜”与“允许短 schedule”解耦：中等轨道变化
时仍可试用旧 MPS，但走充分的两站点重新优化，并保留失败后的随机冷启动。
这不能仅靠放宽当前 `dmrg_switch_tol` 完成，否则初猜复用和短 schedule 会一起放开。
固定轨道上加严最终验收应先于早期 DMRG 精度放宽。

归属澄清：本地 BAGEL `ZHarrison` 默认 `restarted_=false`，
`zharrison.cc:186–233` 每次 `compute()` 重新生成 guess；`restarted_=true`
来自序列化加载。因此上述 MPS 策略是适合我们 DMRG 的改进，不能说是复制
BAGEL 默认跨宏迭代的 CI 热启动。

### 5. 大体系的内存策略，按实际瓶颈再做

BAGEL `zcassecond_compute.cc:298–323` 分批处理响应收缩；我们的三个二阶块
仍一次读入 RAM，约 `48*nmo²*ncas²` 字节。按当前基组维数估计，仅这三块为：
CaOH（404,16）2.01 GB，UNF3 small（676,20）8.77 GB，large（676,30）19.74 GB。
它们不是程序总内存；还包括 DMRG、其他积分、Krylov 及收缩临时量。

先处理生命周期：`zmc_superci.py:1391` 计算新 `build_operators(...)` 时，
旧 `hop` 仍持有上个轨道点的三个块；新 `gen_g_hop` 又在收敛判断前进行
内存检查和预条件分块对角化。可以先释放旧 Hessian 缓存，并把昂贵准备工作
延迟到确实需要轨道更新时。更大体系再考虑从 HDF5 分块收缩，保持 full ERI，
无需引入 CD/DF；当前 500 GB 算例并不能单凭这三个块就判定必须流式化。

### 本次结论与可重现检查

建议次序：**内层失败恢复 + 单态 CI 验收 → 同轨道点重复计算消除 →
DMRG 初猜/schedule 对照 → 按分子实际剖析决定子空间重启和流式积分**。
更复杂的块预条件、轨道–CI 联合 Newton 暂不优先：新版 F 的二阶每步只用
3–5 次 Hessian 作用，目前没有证据证明更复杂的内层能带来主要收益。

审查时用 `.tmp_superci_debug/bagel_review3_probe.py` 验证了内层失败不重试及
重复 core JK。该探针断言的是修复前行为；修复后的可运行检查已放入下面的正式测试。

运行中的 CaOH/UNF3 是此前版本的作业，不能拿它们的当前进度作为新版或以上
未实施建议的收敛证据。本次没有改动或重启这些作业。

## 第 1、3 项实施结果

按用户要求，只补内层失败恢复和消除重复计算，DMRG/Block2 实现、扫描计划、
初猜/重启门控参数及收敛阈值保持原样。对 `dmrg/` 全部 Python 源码逐个核对
SHA256，确认本轮没有修改。

- adaptive 和 second_order 的内层 `maximum_space`、`linear_dependence`
  在残差有限时缩半径重试，与试探 CASCI 共用原有六次尝试上限。未知失败或
  非有限残差直接退出。没有放宽内层验收，也没有接受未收敛方向。
- 内层失败记为 `orbital_trials[*].stage='orbital_solve'`，实际试探记录为
  `'casci'`；宏记录分别统计 `inner_solver_failures` 与 `rejected_trials`。
  只有内层失败时不调用 CASCI、不改重启状态。已经执行过被拒绝的 CASCI 后
  再耗尽内层重试，则沿用原有已接受点的 CI/RDM/checkpoint 恢复。
- full-ERI 的 `_ERIS.get_jk` 缓存带 core occupation 标记的 JK 请求，保存密度
  副本并精确比较后复用。密度改变即重算；响应及活性密度请求不使用该缓存。
  CASCI、普通/adaptive Super-CI 和二阶梯度均通过此共同入口。
- 梯度的 core 密度直接由 core MO 构造，与 CASCI 一致；删除宏迭代末尾
  无用途的第二次矩阵指数及相关死代码。

相关回归 **93 passed, 4 deselected，21.54 秒**，使用上文相同 pytest 命令。
新增及扩展检查包括：

1. 两个真实内层算符都在半径序列 `0.4 → 0.2 → 0.1 → 0.05 → 0.025`
   恢复收敛，期间只在最后成功方向上调用一次试探 CASCI。
2. 数值失败耗尽六次、未知失败、非有限残差均正确终止，未碰过 CI 时保持其状态。
3. 真实 Block2 检查“能量拒绝后连续内层失败”的恢复，最终轨道、MPS/RDM 能量
   及 checkpoint Hamiltonian 仍一致；原有拒绝/重试检查继续通过。
4. CASCI 与两类轨道算符只做一次底层 core JK，结果与独立 JK 一致；密度变化、
   原位修改与响应请求的缓存行为正确。
5. 一次接受的二阶宏更新只调用一次矩阵指数，原有独立能量/梯度/Hessian
   有限差分及实际方向的二次预测检查继续通过。

正式检查在 `tests/test_superci_transactions.py` 和 `tests/test_zmc_second.py`。
本轮没有重跑完整 F 六态作业，前面的 F 轮数和耗时仍对应 248054/248055 版本。
