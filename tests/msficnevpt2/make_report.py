# SPDX-License-Identifier: GPL-3.0-or-later
"""Rebuild the committed report/evidence from completed, audited native runs."""

import hashlib
import json
import pickle
import re
import subprocess
import time

import numpy as np
from socutils.mrpt import solve_msfic

from .carbon import json_value
from .carbon_reference import DIRECTORY


def matrix(value):
    array = np.asarray(value)
    return array[..., 0] + 1j * array[..., 1]


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def load(path):
    with path.open() as handle:
        return json.load(handle)


def revalidate_solver(data, directory):
    """Audit today's kernel on the completed native IC data, retaining timings."""
    with (directory / "prepared.pkl").open("rb") as handle:
        prepared, _ = pickle.load(handle)
    start = time.perf_counter()
    changes = {}
    for key, previous in list(data["results"].items()):
        result = solve_msfic(
            prepared, ansatz=previous["ansatz"], shift=previous["shift"]
        )
        change = float(np.max(np.abs(result.energies - previous["energies"])))
        assert change < 1e-10
        changes[key] = change
        data["results"][key] = json.loads(
            json.dumps(result.__dict__, default=json_value)
        )
    data["solver_revalidation"] = {
        "spectrum_changes": changes,
        "seconds": time.perf_counter() - start,
        "residual_operator": "raw retained IC matrix before numerical Hermitian averaging",
    }


def make_report():
    directory = DIRECTORY.parent
    repo = directory.parents[1]
    base = load(DIRECTORY / "results.json")
    validation = load(DIRECTORY / "validation.json")
    convergence = load(DIRECTORY / "convergence.json")
    rdm_tight = load(directory / "carbon_rdm_tight_run" / "results.json")
    resources = load(DIRECTORY / "resource_audit.json")
    rdm_precision = load(DIRECTORY / "rdm_precision_verified.json")
    regression = (directory / "regression_final_audited.out").read_text()
    assert re.search(r"88 passed, 5 warnings", regression)
    revalidate_solver(base, DIRECTORY)
    revalidate_solver(rdm_tight, directory / "carbon_rdm_tight_run")
    for m, record in convergence.items():
        revalidate_solver(record["data"], directory / f"carbon_M{m}_run")
        for key, result in record["data"]["results"].items():
            energies, reference = (
                np.asarray(result["energies"]),
                np.asarray(base["results"][key]["energies"]),
            )
            record["comparison"][key] = {
                "spectrum_change": float(np.max(np.abs(energies - reference))),
                "spread_change_cm_inverse": abs(
                    float(np.ptp(energies) - np.ptp(reference))
                )
                * 219474.63137,
            }
    for data in (base, rdm_tight, *(r["data"] for r in convergence.values())):
        assert len(data["reference_audit"]["transition_rdm_oracle"]) == 25
        assert max(data["reference_audit"]["residuals"]) < 1e-9
        assert data["reference_audit"]["reference_spread"] * 219474.63137 < 0.001
        for key, result in data["results"].items():
            spread = np.ptp(result["energies"]) * 219474.63137
            assert spread <= 0.02 or key.startswith("ss_sr")
            assert (
                max(
                    d["maximum_ic_relative_residual"]
                    for d in result["diagnostics"]["classes"].values()
                )
                <= 1e-10
            )
    assert len(validation["rotations"]) == 24 and len(validation["metrics"]) == 8
    for shift in (0.2, 0.0):
        ss = np.ptp(base["results"][f"ss_sr_eta_{shift}"]["energies"])
        ms = np.ptp(base["results"][f"ms_mr_eta_{shift}"]["energies"])
        assert ms * 219474.63137 <= 0.001 and ms < ss / 10
    assert all(r["no_cd_df"] and r["no_kr"] for r in resources["references"].values())
    assert all(
        result["status"] == "passed" for result in validation["metrics"].values()
    )
    input_log = (DIRECTORY / "carbon_input_final.out").read_text()
    assert "Exit status: 0" in input_log
    assert "ms_mr energy for each state" in input_log
    paths = [
        "mrpt/nevpt2_eris.py",
        "mrpt/nevpt2_utils.py",
        "mrpt/spinor_helper.py",
        "mrpt/x2cficnevpt2.py",
        "mrpt/x2cqdscnevpt2.py",
        "mrpt/x2cscnevpt2.py",
        "mrpt/x2cmsficnevpt2.py",
        "dmrg/dmrgci.py",
        "mcscf/zmcscf.py",
        "tests/test_x2cmsficnevpt2.py",
        "tests/msficnevpt2/carbon_reference.py",
        "tests/msficnevpt2/carbon.py",
        "tests/msficnevpt2/carbon_input.py",
        "tests/msficnevpt2/validation.py",
        "tests/msficnevpt2/convergence.py",
        "tests/msficnevpt2/rdm_precision.py",
        "tests/msficnevpt2/resource_audit.py",
    ]
    source_hashes = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths
    }
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    evidence = {
        "git_head": head,
        "source_sha256": source_hashes,
        "baseline": base,
        "reference_rotations_and_metric_scan": validation,
        "M_convergence": convergence,
        "RDM_tightening": rdm_tight,
        "native_npdm_gauge_audit": rdm_precision,
        "resources": resources,
        "conversion_cm_inverse_per_hartree": 219474.63137,
        "regression": "88 passed, 5 expected fault-injection warnings",
        "standalone_input": {
            "log": "carbon_tight_run/carbon_input_final.out",
            "exit_status": 0,
            "energies": [
                float(e) for e in re.findall(r"State \d+ E = (\S+)", input_log)
            ],
            "spreads_cm_inverse": [
                float(e) for e in re.findall(r"spread/cm-1 = (\S+)", input_log)
            ],
        },
    }
    (directory / "carbon_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    audit = base["reference_audit"]
    native = base["results"]
    lines = [
        "# C 原子 one-step X2C-MS-FIC-NEVPT2 验证报告",
        "",
        (
            "本轮完成了正式 DMRG 参考与原生 DMRG transition RDM 的端到端计算。"
            "保留 FIC，仅将 SS-SR 的单参考 IC span 改为 MS-MR 的五参考共同 span，"
            "即可将可见的人工展宽降至本轮数值精度。未平均最终能量、未强制标量化 Heff、"
            "未以 FCI 密度替换 DMRG，也未调用 UC/hybrid。"
        ),
        "",
        (
            "这支持本轮的**多态 IC 参数化**解释；不能推广为“所有 contraction 都不影响"
            "对称性”，也没有在本轮重新解决 F 原子的全部问题。"
        ),
        "",
        "## 1. 协议与文献边界",
        "",
        (
            "参照 Reynolds–Shiozaki, JCTC **15**, 1560–1571 (2019), "
            "[DOI 10.1021/acs.jctc.8b00910](https://doi.org/10.1021/acs.jctc.8b00910), "
            "§4.1/Table 2。论文展宽为 SS-SR **1.467**、MS-MR **0.013 cm⁻¹**；"
            "本轮验证方法结构，不宣称逐位复现四分量计算。"
        ),
        "",
        (
            "C 位于原点，最终 charge=0，无外场，默认点核、球谐壳层。"
            "PySCF ANO 库为 ANO-RCC，明确收缩 `ano@4s3p2d1f`：14s9p4d3f 原始壳层，"
            "30 个空间 AO / 60 个 spinor AO。未取得原论文完整 SI，"
            "精确历史 contraction、C 的冻结细节和阈值不冒充已核实。"
        ),
        "",
        (
            "C4+ 闭壳层 HF 提供初猜；中性 C 做含 SOC 的 X2CAMF SA15-DMRG-SCF，"
            "CAS(4e,8 spinors)，2 个 inactive 1s spinors，`frozen=0`。"
            "X2CAMF 默认含 Gaunt/Breit AMF 有效一体修正，PCC/AOC 关闭；"
            "二电子算符为 full Coulomb。它不等于完整四分量 DC/DB 二电子 Hamiltonian。"
            "CD/DF、KR、轨道重排序均关闭。"
        ),
        "",
        (
            "最低 15 态等权 1/15 用于轨道优化和一次公共 SA-Fock；"
            "PT/MS-MR 只用识别出的五个 1D2 根。以 J²≈6、L²≈5.99998936、"
            "S²≈5.23e-6 及独立 70 维 CAS 子空间 overlap 识别为 Python 根 9–13，"
            "不是盲选最接近的五个能量。只在 core/virtual 内半正则化，active 不变。"
        ),
        "",
        (
            "正式 SA 优化使用 `mc.second_order()` 与项目 DMRGCI；"
            "M0=64，16 线程，16 sweeps，noise=1e-5×2、1e-6×2、0×12，"
            "Block2 Davidson residual-square threshold 全程 1e-24，DMRG tol=1e-13；"
            "Davidson deflation subspace=100，random seed=1234，final_one_site=False。"
            "轨道 E/gradient 阈值为 1e-11/1e-6。最终 gradient 约 5.51e-9。"
        ),
        "",
        "## 2. 原生参考基的完整能量",
        "",
        table(
            ["排序编号", "DMRG-SCF / Eh", "SS-SR η=.2 / Eh", "MS-MR η=.2 / Eh"],
            [
                [
                    str(i),
                    f"{audit['all_energies'][r]:.15f}",
                    f"{native['ss_sr_eta_0.2']['energies'][i]:.15f}",
                    f"{native['ms_mr_eta_0.2']['energies'][i]:.15f}",
                ]
                for i, r in enumerate(audit["roots"])
            ],
        ),
        "",
        "各列分别排序；PT 列是完整 Heff 的本征值，不表示逐行跟踪原始 DMRG root。",
        "",
        table(
            ["计算", "本轮展宽 / cm⁻¹", "论文参照 / cm⁻¹"],
            [["reference", f"{audit['reference_spread'] * 219474.63137:.9g}", "<0.001"]]
            + [
                [
                    key,
                    f"{np.ptp(r['energies']) * 219474.63137:.9g}",
                    ("1.467" if key.startswith("ss_sr") else "0.013")
                    if key.endswith("0.2")
                    else "未作逐位对照",
                ]
                for key, r in native.items()
            ],
        ),
        "",
        (
            "η=0 与 η=.2 是不同的移位振幅计算；非零 η 的全矩阵 norm correction "
            "不等于声称复原 η=0 的解。全部无移位原始能量见本节后附矩阵/"
            "[完整 JSON 证据](carbon_evidence.json)。"
        ),
        "",
        "## 3. 误差合同与原生 NPDM",
        "",
        (
            f"参考真实全局残差最大 {max(audit['residuals']):.6e} Eh，"
            f"Gram defect {audit['gram_defect']:.6e}。精确 CAS 只用于独立验证，"
            "SA15 和目标五态子空间的奇异值均为 1（在误差预算内）。"
        ),
        "",
        (
            f"全部 25 个有序态对的原生 rank 1–4 与同一实际 MPS 的独立 determinant "
            f"operator oracle 比较，最大误差 {max(max(x) for x in audit['transition_rdm_oracle'].values()):.6e}。"
            "rank0 使用实测 overlap。普通/transition 的 trace、粒子数 contraction、"
            "反对称性、反向 adjoint 全部严格通过。"
        ),
        "",
        (
            f"MultiMPS 原始 gauge 的 native NPDM 可出现约 2.2e-10 的误差，"
            f"因此初次严格审计正确拒绝运行。最终使用官方 MPSTools QR，"
            f"实际系数/相位最大变化 {audit['npdm_qr_state_and_phase_defect']:.6e}，"
            "不截断、不换参考态、不做新 CI solve。QR 后恢复 dot=2 再计算 NPDM，"
            "消除了此前 one-site SGF 路径的 MKL 调用错误；没有放宽门限。"
        ),
        "",
        table(
            ["ansatz/shift", "最大 IC 相对残差", "最大恢复缺陷 / Eh", "Heff 本征残差"],
            [
                [
                    k,
                    f"{max(x['maximum_ic_relative_residual'] for x in r['diagnostics']['classes'].values()):.6e}",
                    f"{max(x['energy_restoration'] for x in r['diagnostics']['classes'].values()):.6e}",
                    f"{r['diagnostics']['heff_eigen_residual']:.6e}",
                ]
                for k, r in native.items()
            ],
        ),
        "",
        (
            "认证的是保留 IC span 内的投影方程，不要求 UC 全外部空间残差为零。"
            "最终残差使用厄米性审计后的原始投影矩阵，而非只检查数值厄米化后的矩阵。"
            "显著负 metric、丢弃空间的非零 source、失效 RDM/能量恢复和近奇异分母拒绝验收。"
            "可逆的负激发态分母不裁剪。"
        ),
        "",
        "## 4. 预定义参考基变换与精度扫描",
        "",
        (
            "相位=(-.7,.2,.9,-1.3,.4)，置换=(2,4,0,1,3)，复数 U(5) seeds=12,91,407；"
            "每种基、两种 ansatz、η=.2/0 均计算并保留完整矩阵。"
            "旋转后的非对角 Href 保留，SS-SR 采用受限投影的耦合 Sylvester 方程，"
            "MS-MR 使用共同空间。没有预先平均微小 reference splitting。"
        ),
        "",
        table(
            [
                "参考基",
                "SS-SR η=.2 / cm⁻¹",
                "MS-MR η=.2 / cm⁻¹",
                "MS-MR covariance / Eh",
            ],
            [
                [
                    name,
                    f"{validation['rotations'][name + '/ss_sr/eta_0.2']['spread_cm_inverse']:.9g}",
                    f"{validation['rotations'][name + '/ms_mr/eta_0.2']['spread_cm_inverse']:.9g}",
                    f"{validation['rotations'][name + '/ms_mr/eta_0.2']['covariance_defect']:.6e}",
                ]
                for name in (
                    "native",
                    "phases",
                    "permutation",
                    "U5_seed_12",
                    "U5_seed_91",
                    "U5_seed_407",
                )
            ],
        ),
        "",
        (
            "所有 24 个旋转/ansatz/shift 结果见 JSON。SS-SR 随一般 U(5) 改变"
            "（不是误把同一个 span 换坐标）；MS-MR 谱及完整矩阵保持协变。"
            "主表重验使用 16 线程，最终旋转扫描使用 8 线程；近简并本征值的末位"
            "有约 1e-14 Eh 的重复对角化/线程舍入差别，远低于验收预算。"
        ),
        "",
        table(
            ["M0", "SS-SR η=.2 / cm⁻¹", "MS-MR η=.2 / cm⁻¹", "MS-MR 谱变化 / Eh"],
            [
                [
                    "64",
                    f"{np.ptp(native['ss_sr_eta_0.2']['energies']) * 219474.63137:.9g}",
                    f"{np.ptp(native['ms_mr_eta_0.2']['energies']) * 219474.63137:.9g}",
                    "0",
                ]
            ]
            + [
                [
                    m,
                    f"{np.ptp(d['data']['results']['ss_sr_eta_0.2']['energies']) * 219474.63137:.9g}",
                    f"{np.ptp(d['data']['results']['ms_mr_eta_0.2']['energies']) * 219474.63137:.9g}",
                    f"{d['comparison']['ms_mr_eta_0.2']['spectrum_change']:.6e}",
                ]
                for m, d in convergence.items()
            ],
        ),
        "",
        (
            "M32/M128 都重新做 SA15-DMRG-SCF、正式 25 对 RDM 和两种移位的 PT。"
            "简并参考基随不同 DMRG 运行改变，因此 SS-SR 的数值变化不能全部称为"
            "bond-dimension 截断误差；其基依赖由上面的固定 U(5) 实验独立证明。"
        ),
        "",
        table(
            ["M0", "原始实际 bond 范围", "QR 实际 bond 范围"],
            [
                [
                    m,
                    f"{min(r['original_actual_bonds'])}–{max(r['original_actual_bonds'])}",
                    f"{min(r['qr_actual_bonds'])}–{max(r['qr_actual_bonds'])}",
                ]
                for m, r in resources["references"].items()
            ],
        ),
        "",
        "requested M 不是准确性证明；这里还测量了参考残差、子空间、RDM 与能量稳定性。",
        "",
        table(
            ["metric atol=rcond", "MS-MR η=.2 / cm⁻¹", "谱变化 / Eh"],
            [
                [
                    str(c),
                    f"{validation['metrics']['ms_mr/' + str(c)]['spread_cm_inverse']:.9g}",
                    f"{validation['metrics']['ms_mr/' + str(c)]['spectrum_change']:.6e}",
                ]
                for c in (1e-10, 1e-11, 1e-12, 1e-13)
            ],
        ),
        "",
        (
            "四个 metric 截断的 rank 和谱均稳定，SS-SR 也全部通过。"
            "RDM cutoff 从 1e-24 收紧至 1e-30，重新生成全部正式 RDM 与 S/F/V："
        ),
        "",
        table(
            ["计算", "最大谱变化 / Eh"],
            [
                [
                    k,
                    f"{np.max(np.abs(np.asarray(r['energies']) - native[k]['energies'])):.6e}",
                ]
                for k, r in rdm_tight["results"].items()
            ],
        ),
        "",
        "## 5. 软件、资源与复现",
        "",
        (
            f"gpu01 的 CPU 上直接运行（不使用 GPU、不经 Slurm）；Python {resources['python']}，"
            f"版本：`{resources['versions']}`。Git HEAD `{head}`，本轮未提交修改的完整源码 SHA "
            "在 JSON 中。未找到任务书引用的 `input_code_manifest.json` 和上传源码快照，"
            "不能宣称完成了与缺失快照的逐 SHA 比较；旧源码的保留与回归则已验证。"
        ),
        "",
        (
            f"共同准备指纹 `{base['preparation']['fingerprint']}`。完整 active Hamiltonian "
            f"指纹 `{resources['references']['64']['hamiltonian_sha256']}`；基组全部指数/系数、"
            "M32/M128 各自指纹及版本也在 JSON 中。"
        ),
        "",
        table(
            ["运行", "wall / s", "峰值 RSS / GiB"],
            [
                [
                    "M64 原生 PT + 全部 RDM/oracle",
                    f"{base['wall_seconds']:.3f}",
                    f"{base['peak_rss_gib']:.3f}",
                ],
                [
                    "24 个旋转 + 8 个 metric 对照",
                    f"{validation['wall_seconds']:.3f}",
                    f"{validation['peak_rss_gib']:.3f}",
                ],
            ]
            + [
                [
                    "M" + m + " 完整 SA-DMRG-SCF + PT",
                    f"{d['data']['wall_seconds']:.3f}",
                    f"{d['data']['peak_rss_gib']:.3f}",
                ]
                for m, d in convergence.items()
            ]
            + [
                [
                    "RDM cutoff=1e-30",
                    f"{rdm_tight['wall_seconds']:.3f}",
                    f"{rdm_tight['peak_rss_gib']:.3f}",
                ]
            ],
        ),
        "",
        (
            "独立 tiny fermionic tests 覆盖八类 S/F/source、非简并左右恢复、完整 shift norm、"
            "单态极限、相位/置换/U(5)、非对角 Href 的物理 IC-span Sylvester oracle、"
            "弱 source、负/空 metric、缺 rank4、损坏 RDM、负可逆/近奇异分母。"
            "能量零点平移不改变两种响应的回归也通过。新旧 SC/FIC/QD-SC 合计 **88 passed**；"
            "5 条 warning 来自旧测试的故障注入，"
            "不是正式 C 审计被放过。Ruff 和 diff 空白检查通过。短顺序输入也已独立运行，exit=0。"
        ),
        "",
        "复现命令（从仓库根目录；SSH 命令将这些命令在 gpu01 执行）：",
        "",
        "```bash",
        "ulimit -s unlimited",
        "ulimit -c 0",
        "export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16",
        ".venv/bin/python -u -m tests.msficnevpt2.carbon",
        "OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -u -m tests.msficnevpt2.validation",
        ".venv/bin/python -u -m tests.msficnevpt2.convergence",
        ".venv/bin/python -u -m tests.msficnevpt2.carbon_input",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_x2cmsficnevpt2.py tests/test_x2cficnevpt2.py tests/test_x2cqdscnevpt2.py tests/test_x2cscnevpt2_wick.py",
        ".venv/bin/python -u -m tests.msficnevpt2.make_report",
        "```",
        "",
        (
            "主参考使用完成的 checkpoint；`carbon.py` 可恢复 `prepared.pkl`，因此不必重算"
            "SCF/DMRG/RDM。首次完整参考日志为 `carbon_tight_run/pt.out`；最终正式 PT 日志为"
            "`pt_qr_fixed.out`，最终旋转日志为 `validation_final.out`；完整 M/RDM 流程日志为 "
            "`convergence.out`、`rdm_tight.out`，当前求解器在其已审计数据上的重验见归档 JSON。"
            "短输入日志为 `carbon_input_final.out`。这些运行目录、checkpoint、scratch、.out/.pkl "
            "被 gitignore；本报告与精简归档 JSON 是可提交的科学证据。"
        ),
        "",
        (
            "**失败/未运行边界：** 初始较松 DMRG 参考残差和原始 NPDM gauge 精度未通过，"
            "相关日志保留；one-site NPDM 的 MKL 故障未被接受；一次 oracle 包装遗漏 work_memory "
            "已修正并重跑。没有未完成的本轮 C 验收计算。未做完整四分量/原文 SI 逐位复现，"
            "也未开展 F 扩展、UC、无4-RDM或新的 SC 多态理论。"
        ),
        "",
        (
            "独立短输入曾在 SS-SR 的 metric 正交化后被厄米性门限拒绝。"
            "最终按数学等价的次序，先在原始 IC 坐标中相减 F-eS，再作正交化，"
            "避免相减两个已经被小 metric 放大的矩阵；没有放宽门限。"
            "能量零点平移回归已通过，全部已完成的 native IC 数据也以最终求解器重验，"
            "相对旧结果的谱变化、原始投影残差、source 尺度及 condition number 在 JSON 中。"
            "独立短输入的最终能量和展宽单独归档，不混入主表的原生参考基结果。"
        ),
        "",
        "## 6. 完整 Heff、shift norm 与 class 矩阵",
        "",
        (
            "下列矩阵为实际原生参考坐标，单位 Eh（norm 无量纲），未去 trace、未抹除小量。"
            "每类完整 correction/norm 矩阵、V†t 总矩阵及所有旋转矩阵均在 JSON；"
            "八类的 metric/残差/谱审计也逐一保留。"
        ),
        "",
    ]
    for key, result in native.items():
        lines += [
            "### " + key,
            "",
            "五个原始本征值：`"
            + ", ".join(f"{e:.15f}" for e in result["energies"])
            + "`。",
            "",
        ]
        for name in ("heff", "shift_norm"):
            lines += [name + ":", "", "```text"]
            for row in matrix(result[name]):
                lines.append("  ".join(f"{z.real:.15e}{z.imag:+.15e}j" for z in row))
            lines += ["```", ""]
        lines += [
            table(
                ["class", "Tr(correction)/5 / Eh", "最大 IC 相对残差", "metric rank"],
                [
                    [
                        k,
                        f"{np.trace(matrix(c)).real / 5:.15e}",
                        f"{result['diagnostics']['classes'][k]['maximum_ic_relative_residual']:.6e}",
                        str(result["diagnostics"]["classes"][k]["metric"]["rank"]),
                    ]
                    for k, c in result["class_corrections"].items()
                ],
            ),
            "",
            "表中的 trace 仅用于诊断，绝未用于替换矩阵或最终能量。",
            "",
        ]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    print("Report and raw evidence written", directory, flush=True)


if __name__ == "__main__":
    make_report()
