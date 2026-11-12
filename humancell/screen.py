"""In-silico knock-out screens.

A CRISPR screen measures how many cells a clone leaves behind. The model equivalent: delete the gene
(state long after editing: no mRNA, no protein), simulate, and count divisions relative to wild type.
Genes that are not expressed, or whose product has no modelled function, cannot change the simulation
and are scored 1.0 without running it (and are reported as such by `modelled_function`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_cell = None


def modelled_function(cell) -> pd.DataFrame:
    """Per gene: in which sub-models does its product act?"""
    g = cell.genes
    out = pd.DataFrame(index=g.index)
    out["symbol"] = g["symbol"].values
    out["expressed"] = cell.p_ref > 0
    met = set(cell.met.gene_ids)
    out["metabolism"] = [x in met for x in g.index]
    mach = np.zeros(cell.n, bool); mach[cell.mach.all_genes] = True
    out["machine"] = mach
    sig = np.zeros(cell.n, bool); sig[cell.sig.gene_idx[cell.sig.gene_idx >= 0]] = True
    out["signalling"] = sig
    tf = np.zeros(cell.n, bool); tf[cell.sig.tf_gene[cell.sig.tf_gene >= 0]] = True
    out["transcription_factor"] = tf
    mod = np.zeros(cell.n, bool); mod[cell.module_genes] = True
    out["kinetic_module"] = mod
    cols = ["metabolism", "machine", "signalling", "transcription_factor", "kinetic_module"]
    out["any_function"] = out[cols].any(axis=1)
    return out


def _init(kw):
    global _cell
    from .cell import VirtualCell
    _cell = VirtualCell(stochastic=False, **kw)


def _one(args):
    symbols, hours = args
    c = _cell.clone()
    c.reset_shared()                 # the metabolic and signalling models are shared between clones
    c.knockout(*symbols)
    c.run(hours)
    s = c.summary()
    c.washout()
    return dict(gene="+".join(symbols), alive=s["alive"], fate=s["fate"], divisions=s["divisions"],
                phase=s["phase"], growth=float(s["growth"]), mu_cap=s["mu_cap"])


def knockout_screen(symbols, hours: float = 72.0, processes: int = 4, dt: float = 300.0,
                    met_interval: float = 3600.0) -> pd.DataFrame:
    """Simulate single-gene knock-outs. Returns fitness = divisions relative to wild type (0 if the cell dies)."""
    from multiprocessing import Pool
    kw = dict(dt=dt, met_interval=met_interval)
    jobs = [((s,) if isinstance(s, str) else tuple(s), hours) for s in symbols]
    with Pool(processes, initializer=_init, initargs=(kw,)) as pool:
        wt = pool.apply(_one, (((), hours),))
        rows = pool.map(_one, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df["fitness"] = np.where(df["alive"], np.minimum(df["divisions"] / max(wt["divisions"], 1), 1.0), 0.0)
    df.attrs["wild_type"] = wt
    return df
