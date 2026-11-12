"""Run many independent in-silico experiments in parallel.

An experiment is a recipe: a list of (method_name, args, kwargs) applied to a fresh cell, e.g.
    [("run", (6,), {}), ("drug", ("trametinib",), {}), ("run", (48,), {})]
A pseudo-step ("init", (), {...}) passes constructor arguments (seed, traits, stochastic ...).
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np


def _run(job):
    name, recipe, kw, want = job
    from .cell import VirtualCell
    kw = dict(kw)
    for method, args, kwargs in recipe:
        if method == "init":                      # per-experiment constructor arguments, e.g. seed or traits
            kw.update(kwargs)
    c = VirtualCell(**kw)
    for method, args, kwargs in recipe:
        if method != "init":
            getattr(c, method)(*args, **kwargs)
    out = dict(name=name, summary=c.summary(), frame=c.frame(), divisions=list(c.divisions),
               death=getattr(c, "death_cause", None), state=dict(c.s), exchange=dict(c.exchange))
    if "mrna" in want:
        out["mrna"] = c.m.copy()
    if "protein" in want:
        out["protein"] = c.p.copy()
    return out


def run_many(experiments: dict, processes: int = 4, want=(), **cell_kwargs) -> dict:
    """experiments: {name: recipe}. Returns {name: result dict} (summary, time-course frame, final state)."""
    jobs = [(name, recipe, cell_kwargs, tuple(want)) for name, recipe in experiments.items()]
    with Pool(processes) as pool:
        res = pool.map(_run, jobs, chunksize=1)
    return {r["name"]: r for r in res}


def treat(method, *args, pre=6.0, post=66.0, **kwargs):
    """Recipe: grow `pre` hours, apply one perturbation, follow for `post` hours."""
    return [("run", (pre,), {}), (method, args, kwargs), ("run", (post,), {})]


def fate_table(results: dict):
    import pandas as pd
    rows = []
    for name, r in results.items():
        s, st = r["summary"], r["state"]
        rows.append(dict(experiment=name, fate=s["fate"], hours=round(s["hours"], 1), divisions=s["divisions"],
                         phase=s["phase"], growth_per_h=round(float(s["growth"]), 4),
                         glucose_uptake=round(-r["exchange"].get("glucose", 0.0), 2),
                         lactate_secretion=round(r["exchange"].get("L-lactate", 0.0), 2),
                         mTORC1=round(st["mtor"], 2), ISR=round(st["isr"], 2), DSB=round(st["D"], 1)))
    return pd.DataFrame(rows).set_index("experiment")
