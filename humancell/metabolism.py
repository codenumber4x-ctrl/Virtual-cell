"""Genome-scale metabolism: Human-GEM constrained by the simulated proteome.

Why this formalism: the stoichiometry of human metabolism is known genome-wide, the kinetic
constants are not. Flux balance analysis needs only the former. To make it respond to the rest
of the cell, every enzyme-catalysed reaction gets a capacity proportional to the current
abundance of its enzyme (evaluated through the gene-protein-reaction rule), calibrated on a
reference flux state that reproduces measured growth and exchange rates:

    |v_r(t)| <= cap_r * f(E_r(t) / E_r(ref)),   cap_r = |v_r(ref)| / SATURATION  (kcat_default * E_r(ref) if the reaction is idle in the reference)
    f(x) = x / RESERVE for x < RESERVE, 1 for RESERVE <= x <= 1, x for x > 1

The apparent kcat implied by cap_r is an *inferred* parameter (reported, never hidden).
Linear programs are solved with HiGHS and warm-started, so a re-solve costs milliseconds.
"""
from __future__ import annotations

import ast
import pickle

import highspy
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data

BIOMASS = "MAR13082"
ATPM = "MAR03964"
POOLS = {"protein": "protein_pool_biomass", "RNA": "RNA", "DNA": "DNA", "lipid": "lipid_pool_biomass",
         "cofactor": "cofactor_pool_biomass", "metabolite": "metabolite_pool_biomass", "glycogen": "glycogen"}

SATURATION = 0.5        # fraction of enzyme capacity used in the reference state
RESERVE = 0.3           # capacity only falls once an enzyme is depleted below this fraction (see machinery.py)
KCAT_DEFAULT = 10.0     # 1/s, used for expressed enzymes that carry no reference flux
N_A = 6.022e23
INF = highspy.kHighsInf

AMINO_ACIDS = ["alanine", "arginine", "asparagine", "aspartate", "cysteine", "glutamine", "glutamate", "glycine",
               "histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "proline", "serine",
               "threonine", "tryptophan", "tyrosine", "valine"]
ESSENTIAL_AA = ["histidine", "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "threonine",
                "tryptophan", "valine"]
# Medium: name -> (concentration mM, maximal uptake mmol/gDW/h). DMEM + 10% serum-like. Uptake maxima are
# generous; the binding limit is normally the abundance of the transporter (enzyme capacity below).
MEDIUM = {
    "glucose": (25.0, 1.40), "O2": (0.2, 2.0), "glutamine": (4.0, 0.30),
    **{aa: (0.4, 0.03) for aa in AMINO_ACIDS if aa != "glutamine"},
    "cystine": (0.2, 0.04), "choline": (0.03, 0.02), "folate": (0.01, 0.002),
    "nicotinamide": (0.03, 0.002), "pantothenate": (0.02, 0.002), "pyridoxine": (0.02, 0.002),
    "riboflavin": (0.001, 0.002), "thiamin": (0.01, 0.002), "inositol": (0.04, 0.01), "biotin": (0.001, 0.001),
    "lipoic acid": (0.001, 0.001), "aquacob(III)alamin": (0.001, 0.0005), "linoleate": (0.02, 0.01),
    "linolenate": (0.01, 0.005), "pyruvate": (1.0, 0.05), "alpha-tocopherol": (0.001, 0.0005),
    "retinoate": (0.001, 0.0005), "gamma-tocopherol": (0.001, 0.0005),
}
INORGANIC = ["H2O", "H+", "Pi", "sulfate", "Na+", "K+", "Cl-", "Ca2+", "Fe2+", "Fe3+", "Mg2+", "zinc", "HCO3-",
             "NH3", "CO2", "Cu2+", "iodide", "selenate"]
DYNAMIC = ["glucose", "glutamine", "O2", "L-lactate"] + [a for a in AMINO_ACIDS if a != "glutamine"]
KM_UPTAKE = {"glucose": 1.0, "glutamine": 0.3, "O2": 0.005}   # mM
# Measured exchange fluxes used to pin the reference state (typical proliferating cancer line,
# Jain et al. 2012 Science CORE profiles; mmol/gDW/h for a 450 pg dry-mass cell)
REFERENCE_EXCHANGE = {"glucose": -0.65, "L-lactate": 1.05, "glutamine": -0.12}
# Non-growth ATP turnover (ion pumps, protein/RNA turnover, cytoskeleton), mmol/gDW/h. Together with the
# 45 mmol/gDW growth-associated ATP of the biomass reaction this gives ~3.4 mmol/gDW/h in total, matching a
# HeLa oxygen consumption of a few fmol/min/cell plus glycolytic ATP.
ATP_MAINT_REF = 2.0


def network() -> dict:
    """Solver-independent representation of Human-GEM (cached)."""
    pkl = data.DATA / "network.pkl"
    if pkl.exists():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    import warnings
    warnings.filterwarnings("ignore")
    import cobra
    m = cobra.io.load_yaml_model(str(data.fetch("Human-GEM.yml")))
    ids = [r.id for r in m.reactions]
    bio = m.reactions.get_by_id(BIOMASS)
    net = dict(
        S=sp.csc_matrix(cobra.util.array.create_stoichiometric_matrix(m, array_type="lil")),
        ids=ids, names=[r.name for r in m.reactions],
        lb=np.array([r.lower_bound for r in m.reactions]), ub=np.array([r.upper_bound for r in m.reactions]),
        subsystem=[r.subsystem if isinstance(r.subsystem, str) else ";".join(r.subsystem or []) for r in m.reactions],
        gpr=[r.gene_reaction_rule for r in m.reactions],
        equation=[r.build_reaction_string(use_metabolite_names=True) for r in m.reactions],
        met_ids=[x.id for x in m.metabolites], met_names=[x.name for x in m.metabolites],
        met_comp=[x.compartment for x in m.metabolites], compartments=dict(m.compartments),
        exchange={next(iter(r.metabolites)).name: ids.index(r.id) for r in m.reactions
                  if len(r.metabolites) == 1 and next(iter(r.metabolites)).compartment == "e"},
        biomass={x.name: (m.metabolites.index(x), c) for x, c in bio.metabolites.items()},
    )
    with open(pkl, "wb") as fh:
        pickle.dump(net, fh)
    return net


def _compile_gpr(rule: str):
    """GPR string -> nested tuple evaluated as AND=min (complex), OR=sum (isozymes)."""
    if not rule:
        return None
    tree = ast.parse(rule.replace(" and ", " & ").replace(" or ", " | "), mode="eval").body

    def walk(n):
        if isinstance(n, ast.Name):
            return n.id
        op = "and" if isinstance(n.op, ast.BitAnd) else "or"
        return (op, walk(n.left), walk(n.right))
    return walk(tree)


def _eval_gpr(node, level):
    if isinstance(node, str):
        return level.get(node, 0.0)
    op, a, b = node
    a, b = _eval_gpr(a, level), _eval_gpr(b, level)
    return min(a, b) if op == "and" else a + b


def _prune_gpr(node, level):
    """Drop complex subunits that are undetected in the reference proteome (the complex evidently works without them)."""
    if isinstance(node, str) or node is None:
        return node
    op, a, b = node
    a, b = _prune_gpr(a, level), _prune_gpr(b, level)
    if op == "and":
        ea, eb = _eval_gpr(a, level), _eval_gpr(b, level)
        if ea <= 0 < eb:
            return b
        if eb <= 0 < ea:
            return a
    return (op, a, b)


def _gpr_genes(node, out):
    if isinstance(node, str):
        out.add(node)
    elif node is not None:
        _gpr_genes(node[1], out), _gpr_genes(node[2], out)
    return out


class LP:
    """min c.x  s.t.  row_lo <= A x <= row_hi, lb <= x <= ub   (HiGHS, warm-started)."""

    def __init__(self, A, lb, ub, cost, row_lo=None, row_hi=None):
        A = sp.csc_matrix(A)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
        lp.col_cost_ = np.asarray(cost, float)
        lp.col_lower_, lp.col_upper_ = np.asarray(lb, float), np.asarray(ub, float)
        lp.row_lower_ = np.zeros(A.shape[0]) if row_lo is None else row_lo
        lp.row_upper_ = np.zeros(A.shape[0]) if row_hi is None else row_hi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.passModel(lp)
        self.n = A.shape[1]

    def bounds(self, idx, lo, hi):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int32))
        self.h.changeColsBounds(len(idx), idx, np.atleast_1d(np.asarray(lo, float)), np.atleast_1d(np.asarray(hi, float)))

    def solve(self):
        """Warm-started simplex; if it stalls (degenerate re-solve) fall back to interior point."""
        h = self.h
        h.setOptionValue("simplex_iteration_limit", 4000)
        h.run()
        if h.getModelStatus() == highspy.HighsModelStatus.kIterationLimit:
            h.setOptionValue("solver", "ipm")
            h.setOptionValue("simplex_iteration_limit", 2147483647)
            h.run()
            h.setOptionValue("solver", "choose")
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None, None
        return h.getInfo().objective_function_value, np.array(h.getSolution().col_value)


def _abs_lp(S, lb, ub, J, target, extra_cost=None):
    """LP minimising sum_j |v_j - target_j| over reactions J (plus optional linear costs)."""
    m, n = S.shape
    k = len(J)
    sel = sp.csc_matrix((np.ones(k), (np.arange(k), J)), shape=(k, n))
    eye = sp.identity(k, format="csc")
    A = sp.bmat([[S, None], [sel, -eye], [sel, eye]], format="csc")
    cost = np.concatenate([np.zeros(n) if extra_cost is None else extra_cost, np.ones(k)])
    row_lo = np.concatenate([np.zeros(m), np.full(k, -INF), target])
    row_hi = np.concatenate([np.zeros(m), target, np.full(k, INF)])
    return LP(A, np.concatenate([lb, np.zeros(k)]), np.concatenate([ub, np.full(k, INF)]), cost, row_lo, row_hi)


class Metabolism:
    """Proteome-constrained Human-GEM with a medium, a calibrated reference state and fast re-solves."""

    def __init__(self, genes: pd.DataFrame, mu_ref: float | None = None, dry_mass_pg: float | None = None,
                 verbose: bool = False):
        self.genes = genes
        self.mu_ref = mu_ref or np.log(2) / data.PHYS["doubling_h"]
        self.dry_mass_g = (dry_mass_pg or data.PHYS["dry_mass_pg"]) * 1e-12
        self.log = print if verbose else (lambda *a, **k: None)
        net = self.net = network()
        self.S, self.ids = net["S"], net["ids"]
        self.idx = {r: i for i, r in enumerate(self.ids)}
        self.n = len(self.ids)
        self.bio, self.atpm = self.idx[BIOMASS], self.idx[ATPM]
        self.exchange = dict(net["exchange"])
        self.lb, self.ub = net["lb"].copy(), net["ub"].copy()

        # ---- medium
        for i in self.exchange.values():
            self.lb[i], self.ub[i] = 0.0, 1000.0
        for name in INORGANIC:
            if name in self.exchange:
                self.lb[self.exchange[name]] = -1000.0
        self.medium = {k: v[0] for k, v in MEDIUM.items()}
        self.medium["L-lactate"] = 0.0
        self.vmax_uptake = {k: v[1] for k, v in MEDIUM.items() if k in self.exchange}
        self.missing_medium = [k for k in MEDIUM if k not in self.exchange]
        for name in self.vmax_uptake:
            self.lb[self.exchange[name]] = self._uptake_bound(name)
        self.lb[self.bio], self.ub[self.bio] = 0.0, 1000.0

        # ---- GPR compilation
        self.gpr = {i: _compile_gpr(rule) for i, rule in enumerate(net["gpr"]) if rule}
        self.rxn_genes = {i: _gpr_genes(v, set()) for i, v in self.gpr.items()}
        self.gene_ids = sorted(set().union(*self.rxn_genes.values()))
        self.gene_rxns = {}
        for i, gs in self.rxn_genes.items():
            for gid in gs:
                self.gene_rxns.setdefault(gid, []).append(i)
        self.lps = []
        self._calibrate()

    # ------------------------------------------------------------------ medium
    def _uptake_bound(self, name):
        c = max(self.medium.get(name, 0.0), 0.0)
        return -self.vmax_uptake[name] * c / (KM_UPTAKE.get(name, 0.02) + c) if c > 0 else 0.0

    def set_medium(self, **conc):
        """Set extracellular concentrations (mM), e.g. set_medium(glucose=0, O2=0.01)."""
        self.medium.update(conc)
        names = [k for k in conc if k in self.vmax_uptake]
        self._push([self.exchange[k] for k in names], [self._uptake_bound(k) for k in names], None)

    def _push(self, idx, lo=None, hi=None):
        """Write reaction bounds to the master arrays and every solver instance."""
        idx = np.asarray(idx, dtype=np.int32)
        if len(idx) == 0:
            return
        if lo is not None:
            self.lb[idx] = lo
        if hi is not None:
            self.ub[idx] = hi
        keep = (idx != self.bio) & (idx != self.atpm)       # these two are owned by each LP
        for lp in self.lps:
            lp.bounds(idx[keep], self.lb[idx[keep]], self.ub[idx[keep]])

    # ------------------------------------------------------------------ calibration
    def enzyme_levels(self, protein: dict) -> dict:
        return {i: _eval_gpr(node, protein) for i, node in self.gpr.items()}

    def _calibrate(self):
        S, n = self.S, self.n
        prot = self.genes["protein"].to_dict()
        self.gpr = {i: _prune_gpr(node, prot) for i, node in self.gpr.items()}
        self.E_ref = self.enzyme_levels(prot)
        off = np.array([i for i, e in self.E_ref.items() if e <= 0], dtype=int)
        self.log(f"{len(off)} of {len(self.gpr)} enzyme-catalysed reactions have no detectable enzyme")

        # 1. reference constraints: measured growth, exchange rates and maintenance ATP
        lb, ub = self.lb.copy(), self.ub.copy()
        lb[self.bio] = ub[self.bio] = self.mu_ref
        lb[self.atpm] = ATP_MAINT_REF
        for name, v in REFERENCE_EXCHANGE.items():
            i = self.exchange[name]
            lb[i] = v
            if v < 0:
                ub[i] = v

        # 2. reactions without detectable enzyme are closed unless the reference state needs them
        _, x = _abs_lp(S, lb, ub, off, np.zeros(len(off))).solve()
        if x is None:
            raise RuntimeError("reference state infeasible: measured exchange rates cannot support growth")
        needed = off[np.abs(x[off]) > 1e-9]
        self.gapfilled = [self.ids[i] for i in needed]
        closed = np.setdiff1d(off, needed)
        self.closed = set(closed.tolist())
        for arr in (lb, ub, self.lb, self.ub):
            arr[closed] = 0.0
        self.log(f"closed {len(closed)}, kept {len(needed)} undetected-enzyme reactions required for growth")

        # 3. reference flux state = parsimonious (minimal total flux) solution
        _, x = _abs_lp(S, lb, ub, np.arange(n), np.zeros(n)).solve()
        self.v_ref = x[:n].copy()
        self.v_ref[np.abs(self.v_ref) < 1e-10] = 0.0

        # 4. enzyme capacities
        conv = 3600.0 / (N_A * self.dry_mass_g) * 1000.0     # copies * kcat[1/s] -> mmol/gDW/h
        self.cap_ref, self.kcat_app = {}, {}
        for i, e in self.E_ref.items():
            if i in self.closed:
                continue
            v = abs(self.v_ref[i])
            # flux-carrying reactions: capacity inferred from the reference flux; idle ones: default kcat
            self.cap_ref[i] = v / SATURATION if v > 1e-9 else KCAT_DEFAULT * (e if e > 0 else 1.0) * conv
            if v > 1e-9 and e > 0:
                self.kcat_app[self.ids[i]] = v / SATURATION / (e * conv)
        self.orig_lb, self.orig_ub = self.lb.copy(), self.ub.copy()
        self.cap_idx = np.array(sorted(self.cap_ref), dtype=np.int32)
        self.cap_arr = np.array([self.cap_ref[i] for i in self.cap_idx])
        self.eref_arr = np.array([self.E_ref[i] for i in self.cap_idx])
        self.scale = np.ones(len(self.cap_idx))
        self.cap_pos = {int(i): k for k, i in enumerate(self.cap_idx)}
        self.lb[self.cap_idx] = np.maximum(self.orig_lb[self.cap_idx], -self.cap_arr)
        self.ub[self.cap_idx] = np.minimum(self.orig_ub[self.cap_idx], self.cap_arr)

        # 5. solver instances: growth capacity, ATP capacity, minimal-adjustment flux state (linear MOMA,
        #    Segre et al. 2002): under any perturbation the reported flux state is the feasible state
        #    closest to the reference state.
        c = np.zeros(n); c[self.bio] = -1.0
        self.lp_bio = LP(S, self.lb, self.ub, c)
        self.lp_bio.bounds(self.atpm, ATP_MAINT_REF, self.ub[self.atpm])
        c = np.zeros(n); c[self.atpm] = -1.0
        self.lp_atp = LP(S, self.lb, self.ub, c)
        self.active = np.where((np.abs(self.v_ref) > 1e-6))[0]
        self.active = self.active[(self.active != self.bio) & (self.active != self.atpm)]
        secr = np.zeros(n)
        ex = np.array(list(self.exchange.values()))
        secr[ex[self.lb[ex] >= 0]] = 1e-3          # mild cost on secreting anything the reference does not
        self.lp_moma = _abs_lp(S, self.lb, self.ub, self.active, self.v_ref[self.active], extra_cost=secr)
        # biomass-precursor diagnostics: one demand column per pool
        cols, self.pool_cols = [], {}
        for k, (label, name) in enumerate(POOLS.items()):
            row, coef = self.net["biomass"][name]
            cols.append(sp.csc_matrix(([-1.0], ([row], [0])), shape=(S.shape[0], 1)))
            self.pool_cols[label] = (n + k, abs(coef))
        self.lp_pool = LP(sp.hstack([S] + cols, format="csc"), np.concatenate([self.lb, np.zeros(len(cols))]),
                          np.concatenate([self.ub, np.zeros(len(cols))]), np.zeros(n + len(cols)))
        self.lp_pool.bounds(self.bio, 0.0, 0.0)
        self.lp_pool.bounds(self.atpm, ATP_MAINT_REF, self.ub[self.atpm])
        self.lps = [self.lp_bio, self.lp_atp, self.lp_moma, self.lp_pool]
        self.atp_maint = ATP_MAINT_REF
        obj, x = self.lp_bio.solve()
        self.mu_max_ref, self.v_opt = -obj, x
        self.log(f"reference: mu={self.mu_ref:.4f}/h, mu_max={self.mu_max_ref:.4f}/h")

    # ------------------------------------------------------------------ dynamic coupling
    def _apply_scale(self, pos, scale):
        pos = np.asarray(pos, dtype=int)
        idx = self.cap_idx[pos]
        cap = self.cap_arr[pos] * scale
        self.scale[pos] = scale
        self._push(idx, np.maximum(self.orig_lb[idx], -cap), np.minimum(self.orig_ub[idx], cap))

    def update_enzymes(self, protein: dict, tol: float = 0.03, genes=None) -> int:
        """Rescale reaction capacities to the current proteome. Returns the number of bounds changed."""
        if genes is None:
            pos = np.arange(len(self.cap_idx))
        else:
            pos = np.array(sorted({self.cap_pos[r] for g in genes for r in self.gene_rxns.get(g, [])
                                   if r in self.cap_pos}), dtype=int)
        if len(pos) == 0:
            return 0
        e = np.array([_eval_gpr(self.gpr[i], protein) for i in self.cap_idx[pos]])
        x = np.where(self.eref_arr[pos] > 0, e / np.maximum(self.eref_arr[pos], 1e-30), 1.0)
        s = np.where(x < RESERVE, x / RESERVE, np.maximum(x, 1.0))      # tolerant to depletion, responsive to induction
        old = self.scale[pos]
        ch = (np.abs(s - old) > tol * np.maximum(old, 1e-6)) | ((s == 0) != (old == 0))
        if ch.any():
            self._apply_scale(pos[ch], s[ch])
        return int(ch.sum())

    def solve(self, mu_demand: float | None = None, atp_demand: float | None = None, fluxes: bool = True) -> dict:
        """Three warm-started LPs: ATP capacity, growth capacity given maintenance, minimal-adjustment fluxes."""
        demand = self.atp_maint if atp_demand is None else atp_demand
        obj, _ = self.lp_atp.solve()
        atp_cap = -obj if obj is not None else 0.0
        atp_met = min(demand, atp_cap * 0.999)
        self.lp_bio.bounds(self.atpm, atp_met, self.ub[self.atpm])
        obj, _ = self.lp_bio.solve()
        mu_cap = max(-obj, 0.0) if (obj is not None and atp_cap >= demand) else 0.0
        mu = max(min(mu_cap, mu_demand if mu_demand is not None else mu_cap), 0.0)
        out = dict(mu_cap=mu_cap, mu=mu, atp_cap=atp_cap, atp_demand=demand, exchange={}, v=None)
        if fluxes:
            self.lp_moma.bounds([self.bio, self.atpm], [mu * 0.999, atp_met], [max(mu, 1e-9), self.ub[self.atpm]])
            _, x = self.lp_moma.solve()
            if x is not None:
                out["v"] = x[:self.n]
                out["exchange"] = {k: float(x[i]) for k, i in self.exchange.items() if abs(x[i]) > 1e-9}
        return out

    def step_medium(self, exchange: dict, dt_h: float, gdw_per_l: float):
        """Deplete / accumulate medium metabolites given exchange fluxes (mmol/gDW/h)."""
        upd = {}
        for name in DYNAMIC:
            if name in exchange and name != "O2":      # O2 is re-equilibrated with the gas phase
                upd[name] = max(self.medium.get(name, 0.0) + exchange[name] * gdw_per_l * dt_h, 0.0)
        self.set_medium(**upd)

    # ------------------------------------------------------------------ analyses
    def pool_capacities(self) -> dict:
        """Growth rate (1/h) that the synthesis capacity of each biomass precursor pool could sustain alone."""
        out = {}
        for label, (col, coef) in self.pool_cols.items():
            self.lp_pool.bounds(col, 0.0, 1000.0)
            self.lp_pool.h.changeColCost(col, -1.0)
            obj, _ = self.lp_pool.solve()
            out[label] = (-obj if obj is not None else 0.0) / coef
            self.lp_pool.h.changeColCost(col, 0.0)
            self.lp_pool.bounds(col, 0.0, 0.0)
        return out

    def knockout(self, gene_ids, protein: dict | None = None, pools: bool = False):
        """Maximal growth (relative to the reference optimum) after deleting genes; state is restored."""
        level = dict(protein) if protein is not None else self.genes["protein"].to_dict()
        for gid in gene_ids:
            level[gid] = 0.0
        saved = self.scale.copy()
        self.update_enzymes(level, genes=gene_ids, tol=0.0)
        ch = np.where(self.scale != saved)[0]
        hit = np.any(np.abs(self.v_opt[self.cap_idx[ch]]) > self.cap_arr[ch] * self.scale[ch] + 1e-9)
        if hit:
            self.lp_bio.bounds(self.atpm, self.atp_maint, self.ub[self.atpm])
            obj, _ = self.lp_bio.solve()
            val = max(-obj, 0.0) if obj is not None else 0.0
        else:
            val = self.mu_max_ref
        res = val / self.mu_max_ref
        if pools:
            res = (res, self.pool_capacities() if hit else None)
        if len(ch):
            self._apply_scale(ch, saved[ch])
        return res

    def subsystem_fluxes(self, v) -> pd.Series:
        return pd.Series(np.abs(v[:self.n])).groupby(pd.Series(self.net["subsystem"])).sum().sort_values(ascending=False)
