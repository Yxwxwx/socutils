# socutils：与指定 block2 `uno.py` 接口兼容的 complex-spinor PM 局域化

## 0. 项目定义与依据

**目标**：在 X2C/X2CAMF-HF 收敛、活性轨道已经选定后，调用一次与参考文件同签名、同返回结构的 `sort_orbitals()`，得到局域化轨道，然后按原设置启动 DMRG-SCF。实现参考文件的 simple 和 occupation-split 两个分支；不把其中某一个分支硬编码成唯一流程。

**本项目不改变** Hamiltonian、基组、电子数、活性空间张成空间、DMRG 的 M、电子根数、SA 权重、DMRG 排序策略及最终收敛门槛。局域化是一次初始轨道表示选择，不是每个宏迭代内增加的新优化步骤。

### 0.1 唯一算法基线

以用户本次上传的 `uno.py` 为基线，不再以此前项目书或 GitHub 当前 master 的推测行为为准。

- 文件：`uno.py`
- SHA-256：`613e77442c33a0feea57f8b90951356f8c0345af0701ffe306b8495b60ff4783`
- `pmloc`：第 44–190 行。
- `get_uno`：第 193–405 行，仅用于区分流程，本项目不移植这一路。
- `sort_orbitals`：第 446–557 行，主移植目标。
- 文件头声明 GPL-3.0-or-later，并列有 Huanchen Zhai、Zhendong Li、Qiming Sun 的来源信息；派生代码保留相应许可和归属。

项目书中标为“参考行为”的内容来自此文件；标为“spinor 适配”“修正”或“验收要求”的内容是本项目新增设计，不应反过来声称原文件已经实现了它们。

### 0.2 仅用于核对 spinor AO 接口的外部依据

PySCF 官方源码提供 `Mole.aoslice_2c_by_atom`（`offset_2c_by_atom` 的别名）、`mol.nao_2c()`、`int1e_ovlp_spinor`、`int1e_r_spinor`。实施时还应与本地安装版本和 `SpinorSCF.get_ovlp()` 核对。

- `https://pyscf.org/_modules/pyscf/gto/mole.html`
- `https://pyscf.org/_modules/pyscf/scf/dhf.html`

这些资料只用于核对 AO 表示/API，不替换上传文件的局域化定义。

## 1. 从源码确定的行为：必须先锁定

### 1.1 `sort_orbitals` 不是 `get_uno`

`sort_orbitals` 只对 `cas_list` 指定的 active 轨道执行 PM。它不局域化 core，不局域化 external，也不对 external 做 SCDM。非 active 列只在最终拼接时按稳定的原始顺序放入 core/virtual。

`get_uno` 才包含：从 UHF alpha/beta 密度生成 UNO、基于自然占据选空间、core 和 active 分别 PM、external 做 SCDM。不得把这两条调用路径合并成一个自创的“全部轨道分块 PM”。

依据：`uno.py:193–284, 311–360, 446–557`。

### 1.2 默认是 simple 分支，但默认不执行 PM

原签名的 `do_loc=False`；仅设置 `cas_list` 而不设置 `do_loc=True`，不会调用 PM。

当 `split_low == 0.0 and split_high == 0.0`：

1. 按调用者提供的 `cas_list` 次序提取全部 active 列。
2. 仅当 `do_loc=True` 时，对整个 `actmo` 调用一次 `pmloc`。
3. 对整个 active 块执行 `psort`，按新的对角占据降序排列。

因此，对 NUF3 的 30 条 active spinors，`do_loc=True` 且默认 split 参数时，**全部 30 条一起 PM**，允许初始 HF 占据与未占据 active 轨道混合。

依据：`uno.py:446–492`。

### 1.3 显式 split 才做 active 内部占据分组

只要 split 参数不是同时为零，就进入另一个分支，要求 `do_loc=True`、`split_high >= split_low`。

分组严格采用输入时的 active 占据：

```python
lidx = actocc <= split_low
midx = (actocc > split_low) & (actocc <= split_high)
hidx = actocc > split_high
```

分别在每个非空组内 PM、组内 `psort`，然后写回其原有 mask 位置。

重要：**不是把输出一律重排成 low + mid + high，也不再把三组混起来做一次 PM 或全 active 占据排序。** 保留参考代码的 mask 写回语义。

对 HF 的 0/1 spinor 占据，若显式使用 `split_low=0.05, split_high=0.95`，CAS(14e,30s) 才分为 16 条 low、0 条 mid、14 条 high。不得把 14+16 分组写死，也不得把它当作默认。

依据：`uno.py:496–533`。

### 1.4 默认 PM 是 Mulliken，不是 Löwdin

`sort_orbitals` 的 PM 调用不传 `iop`，因此采用 `pmloc(..., iop=0)`。

`pmloc` 原定义：

| iop | 参考行为 |
|---|---|
| 0 | Mulliken 布居的二次 PM |
| 1 | Löwdin 布居的二次 PM；`mol` 为分区 list 时假定输入已经在正交基中 |
| 2 | Boys 的三个位置算符，采用相同 Jacobi 对角化框架 |

本项目的 `sort_orbitals` 默认路径必须继续是 **Mulliken PM**。不能改为 PySCF 的其他默认 population、IAO-PM、Boys、SCDM，或四次 PM 目标。

依据：`uno.py:44–110, 119–190, 489–490, 512, 521, 530`。

### 1.5 原文件用实数 Jacobi，不可直接用于一般复数 spinors

原程序采用实数组、`.T` 和实角度 sin/cos。移植应保留同一二次 PM 目标和按轨道对 Jacobi sweep 的算法结构，但把每对实正交旋转推广为复酉旋转。

“方法相同”的准确含义是相同布居、目标函数、分组、Jacobi 成对优化和排序意图；不意味着复数问题可以与实数问题逐位返回同一个旋转矩阵。

## 2. 交付范围与禁止扩展

### 必须交付

- `lo/uno_spinor.py`，含同签名 `sqrtm`、`lowdin`、`pmloc`、`sort_orbitals`。
- `lo/__init__.py` 中最小导出；若已有同名接口，先检查冲突，不覆盖现有行为。
- 测试：API/分支/排序语义、复数 PM、输入计数、小体系精确 CAS 不变性和可运行分子 smoke。
- 一个官方教程式顺序输入：X2CAMF-HF → 已选 CAS → 一次 PM → 原 DMRG-SCF。
- 文档与真实测试报告，列出所有偏离上游行为的修正。
- 保存原始 `uno.py` 为只读测试基线，附哈希及许可。

### 本次不做

- 不实现 `get_uno`、从 UHF 生成 UNO、重新筛选 CAS、external-SCDM。
- 不增加通用 localization result dataclass，不改变五元返回。
- 不新增 CLI 工作流、JSON 状态管理、自动 PEC 跟踪、MPS 轨道变换。
- 不默认做 core/virtual PM，不在 DMRG-SCF 宏/微迭代内重复 PM。
- 不改 DMRG-CI、AH/micro 优化器、MRPT 公式或 integral backend。
- 不新增或自动重跑 Fiedler，不按新的 Fock 本征向量重新 canonicalize。
- 不把“局域化更好”直接视为“DMRG 一定更快”。

## 3. 公共接口与兼容性契约

### 3.1 精确签名

```python
def sqrtm(s): ...
def lowdin(s): ...

def pmloc(mol, mocoeff, tol=1e-6, maxcycle=1000, iop=0, iprint=1):
    # return ierr, u
    ...

def sort_orbitals(
    mol,
    coeff,
    mo_occ,
    mo_energy,
    cas_list=None,
    nactorb=None,
    nactelec=None,
    do_loc=False,
    split_low=0.0,
    split_high=0.0,
    iprint=1,
):
    # return coeff, mo_occ, mo_energy, nactorb, nactelec
    ...
```

不加 `method`、`population`、`blocks`、`kramers`、`reference`、`return_info` 等新关键字。诊断通过现有 `iprint` 输出和私有测试接口完成，不扩展返回值。

### 3.2 Spinor 语义

- `coeff.shape = (mol.nao_2c(), nmo)`；输入是当前 socutils 使用的 2c spinor AO 表示，不是 4c 大小分量堆叠，也不是未声明的 scalar-alpha/beta 行堆叠。
- 不从数组是否 complex 判断 AO 表示；实值特例仍可属于 spinor 基。
- `mo_occ` 为单 spinor 对角占据，范围为 0–1（只允许浮点容差）。
- `nactorb` 是 individual spinor 数，不是空间轨道数或 Kramers pair 数。
- `nactelec` 是活性电子数。
- `cas_list` 明确为 **0-based** 列号，沿用原函数的 NumPy 索引语义。
- `ncore = mol.nelectron - nactelec`；不除以 2，不要求这个数必为偶数。Kramers 配对是另外的约束，不应混进 general-spinor 计数。
- `cas_list is None` 时要求给出 `nactorb, nactelec`，据此取连续窗口。
- `cas_list` 给定时，以其长度及占据迹确定返回的活性尺寸和电子数；若又给出显式尺寸，要求与结果一致，而不是静默忽略冲突。
- 对当前 HF 用例，应确认补集按原顺序分出的 core 是占据态、external 是未占据态；不能把任意乱序非 active 输入误当作已经满足该前提。

这里追求的是 block2 **接口与有效流程兼容**，而不是把空间轨道 0–2 计数原封不动移给 spinors。

### 3.3 阈值不自动换算

保留 split 默认值 `0.0, 0.0`，不自动改成 `0.05, 0.95`。

两个阈值均须有限；进入 split 时要求 high>=low。按用户给出的数值直接比较，不把大于 1 的阈值偷偷减半，也不因为看到 0/1 占据就自动开启 split。文档可以提示占据范围与阈值含义，但不代替用户改变分支。

### 3.4 返回与副作用

- `pmloc` 返回 `(ierr, u)`，`u.shape=(n_selected,n_selected)`；`ierr=0` 表示达成该算法的 sweep 收敛判据，1 表示迭代未收敛。非法输入抛清楚异常。
- `sort_orbitals` 返回五元组，系数与两列元数据始终一一对应。
- 原函数会在成功调用过程中改写输入数组的 active 位置。为保持该副作用契约，spinor 版本可在工作副本上完成后，再将 active 位置结果提交回调用者数组；随后返回最终 C/A/V 重排后的数组。
- 示例必须传入 `.copy()`，避免污染原始 HF 对象。
- 对需要复数结果而输入为实 dtype 的情况，不得原地回写造成虚部丢失；应提前明确检查 dtype 可写回条件，或在文档列为不回写实输入的必要例外。生产 X2C 路径要求 complex128 系数数组。
- `mo_occ`、`mo_energy` 返回值应为验证虚部可忽略后的实数组。

## 4. PM 布居与目标：只做必要的复数适配

### 4.1 AO overlap 与原子分区

采用与当前 X2C-HF 对应的 spinor overlap：

```python
S = mol.intor_symmetric("int1e_ovlp_spinor")
atom_slices = mol.aoslice_2c_by_atom()
```

核对 S 与 `mf.get_ovlp()` 一致、AO 行数正确。若该仓库使用另一种 AO 表示，应先建立明确适配，不能随便截断、补零或将 scalar overlap 重复两份。

原文件按 `nctr*(2*l+1)` 手工累计 scalar AO，不能照搬。使用每个 atom slice 的最后两个字段构造 AO 分区，验证无遗漏/重复并覆盖全部 spinor AO。

`mol` 为 list 时保留 `iop==1` 限制；list 是行分区，输入已正交化，无需再取 overlap。检查行集合有效，测试可允许分区内非连续索引。

### 4.2 Mulliken PM：默认生产路径

对正在局域化的块 C_B，令 P_A 选择原子 A 的 AO 行：

    Q_A = 1/2 C_B† (S P_A + P_A S) C_B.

可先构造 `T=C_B.conj().T @ S`，再取

    V_A = T[:, rows_A] @ C_B[rows_A, :]
    Q_A = (V_A + V_A†)/2.

保留完整复数非对角元，只在检查后取对角元实部。

应满足：

    Q_A† = Q_A;
    sum_A Q_A = I.

Mulliken 布居不必在 0–1 内，也不必半正定；不能拿 Löwdin 的正性测试错误拒绝正常 Mulliken 输入。

### 4.3 `iop=1/2` 保留原含义，但不替换默认路线

Löwdin：`B=S^(1/2) C_B`，`Q_A=B_A† B_A`。原文件的 `s12.T @ c` 在一般复数 S 下不能机械保留，应使用正确的 Hermitian 正平方根。

list partition：直接使用 `Q_A=C_A† C_A`，因为输入已经在正交表示。

Boys：使用相同 spinor AO 表示的三个 Hermitian 位置积分 `int1e_r_spinor`。只保留 `pmloc(iop=2)` 的既有能力，不将它设为 `sort_orbitals` 的默认。

`iop=2` 的单电子坐标仅服务局域化，不声称它就是含完整 picture-change 的相对论物性算符。

### 4.4 目标函数

对于 PM 的原子矩阵：

    L(U) = sum_A sum_i [(U† Q_A U)_ii]^2.

这是上传文件二次 PM 目标的复数推广。对 Hermitian Q，括号内应为实数。不能把 `Q_A.real` 当作输入，也不能把有虚部的非对角 Q_ij 直接平方代替其正确复数贡献。

`iop=2` 把 A 替换为 x/y/z，继续沿用参考代码的对角位置平方目标。

## 5. 复数 Jacobi：保留成对优化，不换一套局域化方法

### 5.1 实数回归路径

实输入、实观测矩阵的分支保留参考文件的轨道对评分、降序访问、实角度公式和 sweep 收敛语义。对正常非退化样例，与原 `pmloc` 做直接回归。

允许符号/列置换及退化最优方向差别时，比较目标、子空间和收敛，而非强求不唯一的轨道逐元素相同。也要设置非退化、确定性的小例子，锁定排序语义。

### 5.2 一般复数两轨道子问题

对一对列 (i,j)，每个 Hermitian 2×2 子块写为

    Q_A^(ij) = t_A I + v_A · sigma,
    v_A = (Re Q_Aij, -Im Q_Aij, (Q_Aii-Q_Ajj)/2).

构造实对称 3×3 矩阵

    G = sum_A v_A v_A^T.

选择其最大本征值对应的单位向量 n；构造满足

    U_ij sigma_z U_ij† = n · sigma

的 SU(2) 旋转，使 `C[:,[i,j]] <- C[:,[i,j]] @ U_ij`。

当选择 n_z>=0 的等价符号后，可使用

    c = sqrt((1+n_z)/2)
    d = (n_x+i*n_y)/(2*c)
    U_ij = [[c, -d*], [d, c]].

数值上对退化的最大本征空间，选择最接近原 z 方向的最优方向；避免在目标已平坦时引入任意列交换。实现时检查范数、酉性和符号约定，不盲抄公式。

本次更新的目标增量应为

    Delta L = 2 (lambda_max(G) - G_zz).

这一公式来自当前二次 PM 目标的两列精确极大化，是本项目的复数适配推导，不是上传源码里已有的实现。

### 5.3 Sweep 规则与收敛

- 继续按参考的耦合评分大小排序轨道对；复数评分可以采用原复合量的模长：`abs(sum_A Q_Aij*(Q_Aii-Q_Ajj))`，在实数极限回到原评分。
- 每次只更新观测矩阵的相应两行/两列和累计 U，不重新计算 AO population。
- 空空间/单列空间直接返回单位变换与 `ierr=0`。
- `maxcycle` 必须为正整数；不能让 `delta` 未定义。
- 主要停止条件继续为一次完整 sweep 的目标增量小于 `tol`。用真实重算目标核对累计 delta，防止漂移。
- 每个接受的 pair 更新不得显著降低目标；输出最大允许 pair gain/梯度作为诊断，但不伪称求到了全局最优。
- 不引入随机多启动、Boys 预局域化、IAO 或另一个优化目标来“改善”默认结果。
- `pmloc` 保留 ierr 契约；`sort_orbitals` 在 PM 未收敛时抛出说明具体组的异常，而不是忽略 ierr。此为明确记录的安全性差异。

## 6. `psort` 与元数据：保持上游定义，不重新对角化

### 6.1 一体密度和能量代理

上传源码中 `pav` 的 0.5 与 `psort` 的 2.0 配套抵消。spinor 实现采用等价、明确的约定：

    D_AO = C0 diag(n0) C0†
    F_proxy = C0 diag(e0) C0†.

注意 F_proxy 是源码构造的一体能量代理，不直接称为 AO Fock 矩阵。对一个待排序块 X：

    n_X = diag(X† S D_AO S X)
    e_X = diag(X† S F_proxy S X).

由于 X 完全来自原始轨道空间，也可用完全等价而更便宜的形式：

    T = C0† S X
    n_X = diag(T† diag(n0) T)
    e_X = diag(T† diag(e0) T).

这种优化只能在测试证明与上述源定义一致后采用；保存调用前的 C0/n0/e0，不能边覆盖输入边改变代理算符。

### 6.2 排序

- 只使用 `np.argsort(-n_X)` 的降序占据排序语义。
- `e_X` 跟随同一 permutation，不添加能量二次排序，不对 F_proxy 对角化。
- simple 对整个 active 排序；split 只在各个固定 mask 内排序。
- HF split 组内占据理论上相同，返回次序在浮点并列处未必唯一；不要因此引入无说明的新排序法。
- 若未来改成稳定排序或给并列定义 tie-breaker，应单独说明与上传版本的差别，不在首版偷偷加入。

### 6.3 原 HF 密度与输出占据的含义

simple PM 可以混合 active occupied/virtual，输出 `mo_occ` 只是原密度在新基底的对角元，通常不是自然占据数。

原 HF 态在新基底中的密度一般含非对角元：

    D_MO,new = U† diag(n0) U.

不要用 `C_new diag(n_new) C_new†` 冒充原 HF 密度；不要把新对角占据四舍五入后再跑 HF。活性电子数由 active 密度迹决定，不由“排在前面的几条看起来占据”决定。

输出 `mo_energy` 也为原能量代理的对角期望值，不是重新 canonicalize 后的本征值。

## 7. 最终 CAS 重排：有效行为兼容，显式修正原文件瑕疵

### 7.1 保留次序规则

- 初始提取使用调用者的 `cas_list` 次序；不要提前排序改变 split 的组内槽位。
- 按源码把已处理的 active 列/占据/能量写入 `sorted(cas_list)` 的位置。
- 按升序获得未选列 `idx`。
- 返回 `[idx[:ncore], sorted(cas_list), idx[ncore:]]` 的 C/A/V 排列。
- 对连续 NUF3 窗口，这个最终外层排列原本就是恒等；对任意非连续列表则不是。

### 7.2 必须记录的差异清单

| 编号 | 上传文件行为 | 本项目处理 |
|---|---|---|
| D1 | scalar overlap、scalar AO 分区、实转置 | 改为与输入一致的 2c overlap/分区、Hermitian conjugation |
| D2 | `ncore=(N-nactelec)//2`、偶数断言 | individual spinor 用 `ncore=N-nactelec`；不要求 general 模式偶数 |
| D3 | 实正交 Jacobi | 同一目标的复酉 Jacobi，实数分支回归 |
| D4 | `len(actmo[:, mask])` 检查的是 AO 行数，空组也可能进入调用 | 按列数/`np.any(mask)` 判断，空组直接跳过 |
| D5 | 最后只重排 coeff 与 mo_occ，未同步最终重排 mo_energy | 三个返回数组使用同一最终 permutation；加非连续 cas_list 回归 |
| D6 | sort_orbitals 忽略 pmloc 的 ierr | 保留 pmloc 返回结构；wrapper 对未收敛显式报错 |
| D7 | 边界参数/电子数部分靠 assert 或直接 round | 明确校验 shape、finite、index 和电子数一致性，不用 -O 可移除的 assert 作为生产检查 |

不得写“与原文件逐位完全一致”。应写“same public call/return interface and valid branch semantics, with documented complex-spinor adaptations and metadata/edge-case fixes”。

电子数推断：以局域化前后不变的 active 迹确定 `nactelec`，在小容差内取整数；如果差异明显，报错而不是自动掩盖一个不完整的活性空间。若用户显式 nactelec 与迹不同，报具体数值。

## 8. CD、SA 与 Kramers 的边界

### CD

局域化只使用轨道和一体 overlap/population，不依赖 full ERI 或 CD factors。因此同一输入轨道下，full/CD 来源不改变该函数。后续 DMRG-SCF 沿用原 integral route，PM 不触发 AO2MO/DF/CD 重建。

### SA

函数不接收 CI/MPS/root 信息，不应检查 nroots==1。返回一套共同轨道，之后可以用于 SS 或 SA；不得给每个 root 独立局域化后再平均不同基底的 RDM。SA 权重不属于这次接口。

### Kramers restriction：不得伪称参考源码已支持

上传源码没有 Kramers 条件，也没有 `kramers` 参数。一般 complex Jacobi 只保证酉性，不自动保持某个显式列配对。

首版主接口实现一般 complex-spinor PM，不暗中从输入轨道“看起来配对”推断并改变优化流形。后续若需要 KR-preserving PM，必须单独明确扩展范围（例如另一个导入路径、保持同签名），使用保持时间反演的成对/辛旋转并单独测试；不能只事后把 U 与时间反演平均，也不能宣称该约束算法就是原始实 Jacobi 的逐行等价。

在本项目的集成测试中，不要静默关闭用户既有 KR 开关。需要 KR 的调用方必须先验证输出配对结构；未实现或验证时明确报告不支持该组合，而非冒充已验证。

同样，不自动保留原 orbital-irrep 标签：若 active 内 PM 混合了不同表示，标签要与实际轨道一致。没有新 symmetry 参数时，不要悄悄给优化加上上游未定义的分组。

## 9. DMRG-SCF 集成：只插入一次调用

### 9.1 默认 simple，全 active 一起局域化

以下是实现完成后的目标用法；这里不声称当前已有这个模块：

```python
from socutils.lo.uno_spinor import sort_orbitals

# 原来的 X2CAMF-HF 已经收敛。
# 固定 NUF3: 126 electrons, CAS(14e,30 individual spinors)。
cas_list = list(range(112, 142))

lo_coeff, lo_occ, lo_energy, nactorb, nactelec = sort_orbitals(
    mol,
    mf.mo_coeff.copy(),
    mf.mo_occ.copy(),
    mf.mo_energy.copy(),
    cas_list=cas_list,
    do_loc=True,
    # 原默认：全部30条active一次PM。
    split_low=0.0,
    split_high=0.0,
)
assert (nactorb, nactelec) == (30, 14)

# 使用原来的 DMRGCI 初始化代码，参数不改变。
mc = zmcscf.CASSCF(mf, ncas=nactorb, nelecas=nactelec)
mc.fcisolver = solver
mc.mo_coeff = lo_coeff

# 沿用现有轨道优化器/最终精度设置。
# 不在下一行之前将 lo_coeff 重新 canonicalize。
mc.second_order()
```

### 9.2 显式 occupation-split 对照

同一调用仅改变：

```python
split_low=0.05,
split_high=0.95,
```

对上述 HF 占据，分别 PM 16 条空轨道和 14 条占据轨道，中间组为空。要将它与 simple 明确区分；不要把该对照变成所有生产输入的默认。

### 9.3 状态与排序

- 在创建初始 MPS/MPO、计算 orbital_ordering 之前完成 PM。
- 现有 `orbital_ordering` 参数保持原设置，局域化不偷偷调用 Fiedler。
- 新试算使用新 scratch/checkpoint，保存已有 ah200 任务，不覆盖它。
- 不把已建立在旧 active 基底上的 MPS 直接宣称为新基底中相同的物理态；本次不开发 MPS 轨道变换。
- 对势能面，各点使用相同参数规则即可开展自动扫描；本次不承诺各点独立局域化产生的 gauge/局部极值必然连续，也不增加手工逐点选轨道。

## 10. 测试与验收

### A. 源码行为锁定

1. 保存原始 uno.py，不编辑基线；核对 sha256。
2. `inspect.signature` 比对公共签名和默认值，返回长度严格是2/5。
3. 通过 spy/mock 确认 `sort_orbitals` 只将 active 传给 pmloc，默认使用 iop=0。
4. `do_loc=False + (0,0)` 不调用 pmloc；`do_loc=True + (0,0)` 全 active 一次调用。
5. 显式 split 只调用非空组：对 HF 14/30 是两次，列数14和16；不混组、不重排成 low+mid+high。
6. 精确边界：occ==low 属于 low，occ==high 属于 mid，超过 high 才属于 high。
7. 测试 active 输入列表非连续、非升序、空/单列子组；确保源码有效次序与本项目声明一致。
8. 输入 cas_list、nactorb/nactelec 两种路径在同一个有效窗口下返回相同物理结果。

### B. 独立复数代数

1. Hermitian S 的平方根/逆平方根；若非正定或无法可靠正交，不自行删 AO、改变用户子空间。
2. 原子分区，Q_A Hermiticity、sum_A Q_A=I；Löwdin 正性与 Mulliken 不强加正性。
3. 真实复数 Q_ij（特别是纯虚非对角）的两轨道优化，验证 3×3 oracle、酉性与目标增量公式。
4. 随机 SU(2) 抽样检查成对子问题的极大值，验证实/虚方向有限差分与 pair stationarity。
5. 正常实数非退化样例与原 pmloc 直接回归；零轨道、单轨道、平坦目标和 maxcycle=1 的 ierr。
6. 随机列相位变化后，目标与物理投影保持等价；不要求局域化全局最优或不唯一轨道逐元素相同。
7. 局域化前后 active 子空间 projector/principal singular values 不变；选定组之间无混合，非 active 列未被旋转。
8. 通过独立 D_AO/F_proxy 重建 lo_occ、lo_energy，测试 active 迹守恒。
9. 非连续 cas_list 专门暴露上游 mo_energy 最终排列遗漏，并验证修正后逐列一致。
10. 输入错误、未收敛和异常路径不输出虚假的成功标识，不部分污染传入数组。

### C. 物理与集成

- 小复数 active Hamiltonian 的 exact CASCI 在局域化前后同谱。变换积分时显式使用 chemist ERI 的第1/3槽位共轭；不要让 oracle 复用同一错误变换代码。
- simple 混合初始占据/未占据时，使用完整变换密度验证 HF 能量；不靠返回占据对角元重建原态。
- 一次真实 X2C 小分子：HF→PM→DMRG-CI/DMRG-SCF；最终准确度以同精度严格求解比较。
- 同一 full/CD 来源的局域化行为只由输入轨道决定，不生成新 ERI。
- SS/SA 都可接收这套轨道；多根测试保持原根数与权重。
- 不要求有限 M 在局域化前后能量完全相等；这是有限 M 表达能力的比较，不是精确 CAS 不变性测试。

### D. NUF3 性能对照

先读取服务器上的实际日志：

`/home/Yxwxwx/new-dmrgscf/unf3/R_1.750/large/ah200/dmrg.out`

模型当前没有这份日志，不得在报告中虚构其 residual、M、熵或耗时；不要由 ah200 目录名推断 M=200。

固定相同 R、CAS(14e,30s)、Hamiltonian、M、schedule、线程与 ordering，比较：

1. no PM（已有实际输入作为基线）。
2. simple PM（原默认split=0/0，全30条）。
3. 可选 occupation-split PM（显式0.05/0.95）。

先做固定轨道 DMRG-CI 判断 sweep 行为；有意义后再比较完整 DMRG-SCF。报告局域化耗时、总耗时、相同精度下 sweep 数、能量轨迹、discarded weight及已有熵诊断。

不要为制造加速结果同时更改 M 或精度。原子 PM 目标对几乎都在同一 U 原子上的轨道可能接近平坦，不能预设效果必然好。

## 11. 实施顺序与完成定义

1. 先写源行为测试和差异清单，得到项目固定的语义。
2. 做 spinor population 与元数据变换；保留 real pmloc 回归。
3. 实现 complex Jacobi、simple/split wrapper，修正空组与 energy 重排。
4. 接入一次性预处理示例，运行 exact-CAS 和分子 smoke。
5. 在服务器上运行 NUF3 单点对照，记录真实结果；无环境时明确未运行，不用mock替代原生通过。

最终交付包括生产模块、最小导出、测试、顺序示例、兼容性/差异说明和原始测试日志。完成定义是“所选 CAS 的原始子空间不变、complex PM 正确、接口和分支符合基线、与原 DMRG-SCF 正常衔接”，不是“必须降低能量或提高速度”。

## 12. 关于旧项目书

本文件覆盖此前项目书中不符合上传源码的假设：

- 删除“sort_orbitals 统一对 core/active/virtual 做 PM”的说法。
- 删除“30条active必须拆成14+16”的默认要求。
- 删除“sort_orbitals 默认 Löwdin”的设计。
- 删除不同返回类型、多方法框架、自动重局域化和自动跟踪作为首版完成条件。
- 保留且落实上传源码真正提供的 simple 与 occupation-split 两个分支。
