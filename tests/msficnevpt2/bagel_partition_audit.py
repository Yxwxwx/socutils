"""Audit occupied MOs and an optional independent small full-chain export."""
import argparse
from pathlib import Path

import numpy as np
from pyscf import lib

from .ag_audit import write_json
from .bagel_reference import MOIntegrals


def audit_sample(full, sample):
    selected = np.asarray(sample.meta["selected_positive_spinors"], dtype=int)
    if (len(selected) != 2 * sample.n or len(np.unique(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= 2 * full.n)):
        raise ValueError("invalid independent MO sample mapping")
    original = full.physicist_block((selected,) * 4)
    independent = sample.physicist_block((np.arange(len(selected)),) * 4)
    h = full.one_body()[np.ix_(selected, selected)]
    if not all(np.isfinite(x).all() for x in (original, independent, h, sample.one_body())):
        raise ValueError("nonfinite native MO sample")
    return dict(selected_positive_spinors=selected.tolist(),
        h1_error=float(np.max(abs(h - sample.one_body()))),
        g2_error=float(np.max(abs(original - independent))),
        g2_pair_exchange=float(np.max(abs(original - original.transpose(1, 0, 3, 2)))),
        g2_hermiticity=float(np.max(abs(original - original.transpose(2, 3, 0, 1).conj()))))


def main(directory):
    lib.num_threads(1)
    full = MOIntegrals(directory / "native_full")
    active = MOIntegrals(directory / "native_active")
    nc, na = 2 * full.meta["nclosed_pairs"], 2 * full.meta["nactive_pairs"]
    no = nc + na
    h = full.one_body()[:no, :no]
    print("NATIVE_AUDIT_H1_READ", flush=True)
    g = full.physicist_block((np.arange(no),) * 4)
    print("NATIVE_AUDIT_G2_READ", flush=True)
    ga = active.physicist_block((np.arange(na),) * 4)
    folded = h + np.einsum("piqi->pq", g[:, :nc, :, :nc]) - np.einsum("piiq->pq", g[:, :nc, :nc, :])
    report = dict(h1_hermiticity=float(np.max(abs(h - h.conj().T))),
        g2_pair_exchange=float(np.max(abs(g - g.transpose(1, 0, 3, 2)))),
        g2_hermiticity=float(np.max(abs(g - g.transpose(2, 3, 0, 1).conj()))),
        active_g2_error=float(np.max(abs(g[nc:, nc:, nc:, nc:] - ga))),
        folded_h1_hermiticity=float(np.max(abs(folded - folded.conj().T))),
        active_h1_error=float(np.max(abs(folded[nc:, nc:] - active.one_body()))),
        reconstructed_core=float(np.real(.5 * np.trace((h + folded)[:nc, :nc]))),
        native_core=active.meta["core_energy"], raw_active_block_errors={})
    for tag, reference in active.g.items():
        block = full.g[tag][(slice(nc // 2, no // 2),) * 4]
        report["raw_active_block_errors"][str(tag)] = float(np.max(abs(block - reference)))
    if (directory / "native_sample/export.json").exists():
        report["independent_sample"] = audit_sample(full, MOIntegrals(directory / "native_sample"))
    write_json(directory / "native_blocks_audit.json", report)
    print("NATIVE_BLOCKS_AUDIT", report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory.resolve())
