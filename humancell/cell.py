"""The integrated virtual cell.

Architecture (after Karr et al. 2012 / Vivarium): independent process models read and write one
shared cell state and are advanced in turn over a short time step. Each process uses the
mathematics its data supports:

  process                              formalism                               genes
  -----------------------------------  --------------------------------------  -----------------
  transcription, RNA decay             stochastic birth-death, per gene        all (~20k)
  translation, protein turnover        stochastic birth-death, per gene        all (~20k)
  molecular machines                   Liebig capacity of curated complexes    ~900
  metabolism                           proteome-constrained FBA (Human-GEM)    ~2.9k
  signalling                           signed network, reference-linearised    ~5k
  gene regulation                      TF-activity regulons (CollecTRI)        ~1.1k TFs
  energy / nutrient sensing, ISR, UPR  reduced kinetic ODEs                    module genes
  hypoxia, redox, DNA damage, p53      reduced kinetic ODEs                    module genes
  ions, membrane potential, calcium    pump-leak ODEs                          module genes
  cell cycle, replication, division    checkpoint-gated hybrid automaton       module genes
  apoptosis / necrosis                 bistable caspase switch, energy/lysis   module genes

Units: copies per cell, hours, mM (medium), mmol/gDW/h (fluxes). "rel" always means relative to
the reference cell at the same size, so every module is at 1.0 in an unperturbed cell.
"""
from __future__ import annotations

import copy
import re

import numpy as np
from scipy.optimize import brentq

from . import data
from .machinery import COMPARTMENT_MACHINE, Machines
from .metabolism import ESSENTIAL_AA, Metabolism
from .signaling import A_REF, Signaling

LN2 = np.log(2.0)

# Cell-line specific lesions that expression data cannot show (HeLa: HPV18 E6 degrades p53, E7 inactivates RB)
TRAITS = {"HeLa": dict(p53_extra_deg=3.0, rb_function=0.3)}

# drug -> ({target symbol: fractional inhibition at dose 1}, {environment parameter: value at dose 1})
DRUGS = {
    "rapamycin": ({"MTOR": 0.9}, {}), "torin1": ({"MTOR": 0.98}, {}),
    "trametinib": ({"MAP2K1": 0.97, "MAP2K2": 0.97}, {}), "erlotinib": ({"EGFR": 0.95}, {}),
    "alpelisib": ({"PIK3CA": 0.95}, {}), "MK2206": ({"AKT1": 0.95, "AKT2": 0.95, "AKT3": 0.95}, {}),
    "palbociclib": ({"CDK4": 0.97, "CDK6": 0.97}, {}), "RO3306": ({"CDK1": 0.95}, {}),
    "nutlin3": ({"MDM2": 0.95}, {}), "bortezomib": ({"PSMB5": 0.95}, {}),
    "cycloheximide": ({}, {"translation_block": 0.97}), "actinomycinD": ({}, {"transcription_block": 0.97}),
    "oligomycin": ({"re:^ATP5": 0.98, "re:^MT-ATP": 0.98}, {}), "rotenone": ({"re:^NDUF": 0.98, "re:^MT-ND": 0.98}, {"ros": 1.0}),
    "2DG": ({"re:^HK\\d$": 0.95, "GCK": 0.95, "ADPGK": 0.95, "GPI": 0.9}, {}), "methotrexate": ({"re:^DHFR": 0.99}, {}),
    "hydroxyurea": ({"RRM1": 0.95, "RRM2": 0.95}, {"replication_stress": 1.0}),
    "5FU": ({"TYMS": 0.95}, {"replication_stress": 0.5}), "simvastatin": ({"HMGCR": 0.97}, {}),
    "etoposide": ({"TOP2A": 0.5}, {"dsb_rate": 60.0}), "doxorubicin": ({"TOP2A": 0.7}, {"dsb_rate": 80.0, "ros": 0.5}),
    "olaparib": ({"PARP1": 0.95}, {"dsb_rate": 3.0}),
    "paclitaxel": ({}, {"spindle_poison": 0.98}), "nocodazole": ({}, {"spindle_poison": 0.98}),
    "tunicamycin": ({"DPAGT1": 0.95}, {"er_stress": 1.0}), "thapsigargin": ({"ATP2A2": 0.98, "ATP2A3": 0.98}, {}),
    "ouabain": ({"re:^ATP1A\\d$": 0.99}, {}), "venetoclax": ({"BCL2": 0.97}, {}), "navitoclax": ({"BCL2": 0.95, "BCL2L1": 0.95}, {}),
    "S63845": ({"MCL1": 0.97}, {}), "H2O2": ({}, {"ros": 2.0}), "CB839": ({"GLS": 0.95}, {}),
    "AZD7762": ({"CHEK1": 0.95, "CHEK2": 0.9}, {}), "KU55933": ({"ATM": 0.95}, {}),
}

APC_ACTIVITY = {"G1": 2.5, "S": 0.2, "G2": 0.2, "M": 0.0}
SCF_ACTIVITY = {"G1": 0.3, "S": 3.0, "G2": 1.5, "M": 1.0}

WATCH = ["TP53", "MDM2", "CDKN1A", "MYC", "CCND1", "CCNE1", "CCNA2", "CCNB1", "MCL1", "BCL2L1", "BAX", "BBC3",
         "HIF1A", "SLC2A1", "HK2", "LDHA", "ATF4", "DDIT3", "HSPA5", "FOS", "JUN", "ACTB", "GAPDH", "RPL3"]


def _sat(x, reserve=0.6):
    """Dose-response of a regulator: tolerant to depletion down to `reserve`, half-sensitive to excess."""
    return x / reserve if x < reserve else (1.0 if x <= 1.0 else 1.0 + 0.5 * (min(x, 3.0) - 1.0))


class VirtualCell:
    def __init__(self, cell_line: str = "HeLa", seed: int = 0, stochastic: bool = True, dt: float = 60.0,
                 met_interval: float = 900.0, metabolism: Metabolism | None = None, signaling: Signaling | None = None,
                 culture_gdw_per_l: float = 0.09, watch=None, traits: dict | None = None):
        g = self.genes = data.build_gene_table(cell_line)
        self.cell_line = cell_line
        self.rng = np.random.default_rng(seed)
        self.stochastic = stochastic
        self.dt = dt                       # s
        self.met_interval = met_interval   # s
        self.traits = TRAITS.get(cell_line, {}) if traits is None else traits
        self.symbols = g["symbol"].values
        self.ids = g.index.values
        self.sym = {}
        for i, s in enumerate(self.symbols):
            self.sym.setdefault(s, i)
        self.n = len(g)

        # ---- per-gene parameters (h^-1)
        self.m_ref, self.p_ref = g["mrna"].values.astype(float), g["protein"].values.astype(float)
        self.k_tx, self.k_tl = g["k_tx"].values.astype(float), g["k_tl"].values.astype(float)
        self.d_m, self.d_p = g["d_m"].values.astype(float), g["d_p"].values.astype(float)
        self.mass = g["mass"].values.astype(float)
        self.length = g["length"].values.astype(float)
        self.mu_ref = LN2 / data.PHYS["doubling_h"]
        self.Mp_ref = float((self.p_ref * self.mass).sum())
        self.expr = self.p_ref > 0

        # ---- sub-models
        self.met = metabolism or Metabolism(g)
        self.sig = signaling or Signaling(g)
        self.mach = Machines(self.symbols, self.p_ref)
        self.met_gene_idx = np.array([self.ids.tolist().index(x) if x in set(self.ids) else -1 for x in []], dtype=int)
        pos = {x: i for i, x in enumerate(self.ids)}
        self.met_genes = [x for x in self.met.gene_ids if x in pos]
        self.met_gene_idx = np.array([pos[x] for x in self.met_genes], dtype=int)
        self.culture_gdw_per_l = culture_gdw_per_l      # 2e5 cells/mL x 450 pg
        comp = g["compartment"].values
        self.is_mt = np.array([s.startswith("MT-") for s in self.symbols])
        self.comp_groups = {c: np.where((comp == c) & ~self.is_mt)[0] for c in COMPARTMENT_MACHINE}
        self.sig_gene = self.sig.gene_idx
        self.sig_has = self.sig_gene >= 0

        # ---- perturbation state
        self.ko = np.zeros(self.n, bool)
        self.tx_mult = np.ones(self.n)          # knock-down / over-expression
        self.func = np.ones(self.n)             # functional fraction (inhibitors)
        self.env = dict(translation_block=0.0, transcription_block=0.0, dsb_rate=0.0, replication_stress=0.0,
                        spindle_poison=0.0, er_stress=0.0, ros=0.0, gpcr=0.0)
        self.drugs = {}

        # ---- molecular state: a newborn G1 cell (0.69 x the cycle-averaged composition)
        birth = LN2
        self.m = self.m_ref * birth
        self.p = self.p_ref * birth
        if stochastic:
            self.m = self.rng.poisson(self.m).astype(float)
            self.p = np.round(self.p)
        self.t = 0.0                            # s
        self.phase = "G1"
        self.q = self.r = self.b = self.mprog = 0.0     # G1 commitment, replicated fraction, G2 progress, mitosis progress
        self.mitotic_h = 0.0
        self.arrested = False
        self.generation = 0
        self.alive = True
        self.fate = "alive"
        self.divisions = []
        # module state (all relative to reference)
        self.s = dict(EC=1.0, supply=1.0, mu_cap=self.met.mu_max_ref, mu=self.mu_ref, mu_dem=self.mu_ref,
                      atp_cap=10.0, atp_demand=2.0, dna_cap=1.0, glyc=1.0, oxphos=1.0,
                      U=0.0, D=1.0, P53=1.0, Mdm2i=1.0, Mdm2=1.0, C=0.0, ros=1.0,
                      Na=12.0, K=140.0, Vm=0.0, swell=1.0, Ca=0.1, Ca_er=400.0,
                      mtor=1.0, isr=0.0, ampk=0.0, autophagy=0.0, low_energy_h=0.0, pump=1.0)
        self._init_modules()
        self.exchange = {}
        self.flux = None
        self._since_met = np.inf
        self.watch = [w for w in (watch or WATCH) if w in self.sym]
        self.history = {k: [] for k in ["t", "phase", "size", "mu", "mu_cap", "EC", "supply", "P53", "D", "C", "U",
                                        "Vm", "Ca", "swell", "mtor", "isr", "ampk", "ros", "r", "mrna_total",
                                        "protein_total", "glucose", "lactate", "q_glc", "q_lac", "q_o2", "q_gln",
                                        "ERK", "AKT", "generation"]}
        self.history.update({f"p:{w}": [] for w in self.watch})
        self.history.update({f"m:{w}": [] for w in self.watch})
        self.record_every = 300.0
        self._since_rec = np.inf

    # ================================================================== helpers
    def _g(self, *names):
        return np.array([self.sym[x] for x in names if x in self.sym and self.p_ref[self.sym[x]] > 0], dtype=int)

    def _init_modules(self):
        s = self.s
        # p53 / Mdm2 parameters, scaled so that (P53, Mdm2) = (1, 1) is the undamaged steady state of a wild-type cell
        self.p53 = dict(d0=0.1, d1=1.3, a=1.4, dm=1.4, K=2.0, n=3.0)
        self.p53["sp"] = self.p53["d0"] + self.p53["d1"]
        self.p53["b"] = self.p53["a"] * (1 + self.p53["K"] ** self.p53["n"])
        extra = self.traits.get("p53_extra_deg", 0.0)
        if extra:                                   # viral E6: settle to the lower steady state
            for _ in range(4000):
                self._p53_step(0.01, 0.0, 1.0, 1.0)
        self.p53_ref = s["P53"]
        # ion homeostasis reference
        self.ion = dict(Na_ref=12.0, K_ref=140.0, J=20.0)        # mM, mM/h pump cycles
        self.ion["Vm_ref"] = self._vm(12.0, 140.0)
        s["Vm"] = self.ion["Vm_ref"]
        # calcium reference fluxes (uM/s)
        self.ca = dict(Vs=1.0, Ks=0.2, Vp=0.3, Kp=0.2, er_ratio=0.02)
        c = self.ca
        js = c["Vs"] * 0.1 ** 2 / (0.1 ** 2 + c["Ks"] ** 2)
        c["k_rel"] = js / (400.0 - 0.1)                     # leak + IP3R release balances SERCA
        c["j_in"] = c["Vp"] * 0.1 / (0.1 + c["Kp"])         # influx balances PMCA
        # gene groups used by the modules
        G = self._g
        self.gg = dict(
            cycD=G("CCND1", "CCND2", "CCND3"), cdk46=G("CDK4", "CDK6"), cycE=G("CCNE1", "CCNE2"), cdk2=G("CDK2"),
            cycA=G("CCNA2"), cycB=G("CCNB1", "CCNB2"), cdk1=G("CDK1"), p21=G("CDKN1A"), p27=G("CDKN1B"),
            rb=G("RB1"), cdc25=G("CDC25A", "CDC25B", "CDC25C"), wee1=G("WEE1", "PKMYT1"), e2f=G("E2F1", "E2F2", "E2F3"),
            tp53=G("TP53"), mdm2=G("MDM2"), atm=G("ATM"), atr=G("ATR"), chk1=G("CHEK1"), chk2=G("CHEK2"),
            bax=G("BAX", "BAK1"), puma=G("BBC3"), noxa=G("PMAIP1"), bim=G("BCL2L11"),
            anti=G("BCL2L1", "MCL1", "BCL2"), xiap=G("XIAP"), dr=G("TNFRSF10B", "TNFRSF10A"), fas=G("FAS"), tnfr1=G("TNFRSF1A"), fadd=G("FADD"),
            casp8=G("CASP8"), cflar=G("CFLAR"), rtk=G("EGFR", "IGF1R", "INSR", "FGFR1", "PDGFRB", "MET"), chop=G("DDIT3"),
            apc_deg=G("CCNB1", "CCNB2", "CCNA2", "PLK1", "AURKA", "CDC20", "PTTG1"),
            scf_deg=G("CCNE1", "CCNE2", "CDT1", "CDC6", "CDC25A"),
        )
        self.module_genes = sorted({int(i) for v in self.gg.values() for i in v})

    def _rel(self, idx, functional, size):
        """Summed functional abundance of a paralogue group relative to reference (1 if the group is absent)."""
        if len(idx) == 0:
            return 1.0
        return float(functional[idx].sum() / (self.p_ref[idx].sum() * size))

    @staticmethod
    def _vm(na, k, p_na=0.05):
        return 61.5 * np.log10((5.0 + p_na * 145.0) / (k + p_na * na))

    # ================================================================== perturbations
    def _idx(self, symbol):
        if symbol not in self.sym:
            raise KeyError(f"unknown gene symbol {symbol}")
        return self.sym[symbol]

    def knockout(self, *symbols, instant: bool = True):
        """Delete genes. instant=True also removes existing mRNA and protein (the state long after a CRISPR edit)."""
        for sname in symbols:
            i = self._idx(sname)
            self.ko[i] = True
            if instant:
                self.m[i] = self.p[i] = 0.0
        self._since_met = np.inf
        return self

    def knockdown(self, symbol, remaining: float = 0.2):
        self.tx_mult[self._idx(symbol)] = remaining
        return self

    def overexpress(self, symbol, fold: float = 5.0):
        i = self._idx(symbol)
        self.tx_mult[i] = fold
        if self.k_tx[i] == 0:                           # ectopic expression of a silent gene
            self.k_tx[i] = np.median(self.k_tx[self.k_tx > 0]) * fold
            self.k_tl[i] = np.median(self.k_tl[self.k_tl > 0])
            self.p_ref[i] = self.k_tx[i] / (self.d_m[i] + self.mu_ref) * self.k_tl[i] / (self.d_p[i] + self.mu_ref)
        return self

    def inhibit(self, symbol, fraction: float = 0.9):
        """Pharmacological inhibition: the protein is present but `fraction` of it is inactive."""
        self.func[self._idx(symbol)] = 1.0 - fraction
        self._since_met = np.inf
        return self

    def drug(self, name: str, dose: float = 1.0):
        """Apply a drug from the DRUGS table; dose 1 = saturating, dose scales target occupancy as d/(d+0.1)*1.1."""
        targets, env = DRUGS[name]
        occ = min(dose / (dose + 0.1) * 1.1, 1.0)
        for sname, frac in targets.items():
            if sname.startswith("re:"):
                rx = re.compile(sname[3:])
                hit = [i for i, x in enumerate(self.symbols) if rx.match(x)]
            else:
                hit = [self.sym[sname]] if sname in self.sym else []
            for i in hit:
                self.func[i] = min(self.func[i], 1.0 - frac * occ)
        for k, v in env.items():
            self.env[k] = max(self.env[k], v * occ)
        self.drugs[name] = dose
        self._since_met = np.inf
        return self

    def washout(self):
        self.func[:] = 1.0
        for k in self.env:
            self.env[k] = 0.0
        self.drugs = {}
        self._since_met = np.inf
        return self

    def set_medium(self, **conc):
        self.met.set_medium(**conc)
        self._since_met = np.inf
        return self

    def serum(self, level: float):
        self.sig.serum(level)
        return self

    def ligand(self, **levels):
        self.sig.set(**levels)
        return self

    def irradiate(self, gray: float):
        """Ionising radiation: ~35 double-strand breaks per Gy per diploid genome."""
        self.s["D"] += 35.0 * gray
        return self

    # ================================================================== modules
    def _p53_step(self, dt_h, atm, mdm2_avail, tp53_expr):
        s, k = self.s, self.p53
        n = max(1, int(np.ceil(dt_h / 0.01)))
        h = dt_h / n
        for _ in range(n):
            P, Mi, M = s["P53"], s["Mdm2i"], s["Mdm2"]
            deg = k["d0"] + self.traits.get("p53_extra_deg", 0.0) + k["d1"] * M * mdm2_avail * (1 - 0.9 * atm)
            hill = P ** k["n"] / (k["K"] ** k["n"] + P ** k["n"])
            s["P53"] = max(P + h * (k["sp"] * tp53_expr - deg * P), 0.0)
            s["Mdm2i"] = max(Mi + h * (k["b"] * hill - k["a"] * Mi), 0.0)
            s["Mdm2"] = max(M + h * (k["a"] * Mi - k["dm"] * M), 0.0)

    def _calcium(self, dt_h, serca, pmca, ip3, ec):
        s, c = self.s, self.ca
        er = s["Ca_er"]
        rel = c["k_rel"] * (0.6 + 0.4 * ip3)
        soce = 1.0 + 4.0 * max(0.0, 1.0 - er / 400.0)

        def net(ca):
            return (c["j_in"] * soce + rel * (er - ca) - c["Vs"] * serca * ec * ca ** 2 / (ca ** 2 + c["Ks"] ** 2)
                    - c["Vp"] * pmca * ec * ca / (ca + c["Kp"]))
        s["Ca"] = brentq(net, 1e-6, 1e4) if net(1e4) < 0 else 1e4
        ca = s["Ca"]
        fill = c["Vs"] * serca * ec * ca ** 2 / (ca ** 2 + c["Ks"] ** 2) - rel * (er - ca)
        s["Ca_er"] = float(np.clip(er + fill / c["er_ratio"] * dt_h * 3600.0 * 0.01, 1.0, 800.0))

    def _ions(self, dt_h, pump_avail, ec):
        s, io = self.s, self.ion
        f = lambda na: (na / (na + 10.0)) ** 3
        n = max(1, int(np.ceil(dt_h / 0.005)))
        h = dt_h / n
        phi = pump_avail * ec / (ec + 0.1) * 1.1
        for _ in range(n):
            j = io["J"] * phi * f(s["Na"]) / f(io["Na_ref"])
            s["Na"] += h * (3 * io["J"] * (145.0 - s["Na"]) / 133.0 - 3 * j)
            s["K"] += h * (-2 * io["J"] * (s["K"] - 5.0) / 135.0 + 2 * j)
        s["pump"] = phi * f(s["Na"]) / f(io["Na_ref"])
        s["Vm"] = self._vm(s["Na"], s["K"])
        depol = np.clip((s["Vm"] - io["Vm_ref"]) / (0.0 - io["Vm_ref"]), 0.0, 1.0)
        s["swell"] = 1.0 + 0.8 * depol

    # ------------------------------------------------------------------ fast exact-enough sampling
    def _poisson(self, lam):
        """Poisson draws; Gaussian limit where the mean is large (>50), exact below."""
        out = np.empty_like(lam)
        big = lam > 50.0
        out[~big] = self.rng.poisson(lam[~big])
        out[big] = np.maximum(np.rint(lam[big] + np.sqrt(lam[big]) * self.rng.standard_normal(int(big.sum()))), 0.0)
        return out

    def _binomial(self, n, prob):
        """Binomial draws; Gaussian limit where n*p > 50, exact below."""
        mean = n * prob
        out = np.empty_like(mean)
        big = mean > 50.0
        out[~big] = self.rng.binomial(n[~big].astype(np.int64), prob[~big])
        sd = np.sqrt(mean[big] * (1.0 - prob[big]))
        out[big] = np.clip(np.rint(mean[big] + sd * self.rng.standard_normal(int(big.sum()))), 0.0, n[big])
        return out

    # ================================================================== one time step
    def step(self):
        if not self.alive:
            return
        s, dt_h = self.s, self.dt / 3600.0
        functional = self.p * self.func
        Mp = float((self.p * self.mass).sum())
        size = max(Mp / self.Mp_ref, 1e-9)
        F = self.mach.factors(functional, size)
        rel = lambda key: self._rel(self.gg[key], functional, size)

        # ---------------- metabolism (slow, every met_interval)
        if self._since_met >= self.met_interval:
            self._metabolism(functional, size, F)
            self._since_met = 0.0
        self._since_met += self.dt
        ec, supply = s["EC"], s["supply"]

        # ---------------- sensing modules
        med = self.met.medium
        aa_s = min(med.get(a, 0.0) / (med.get(a, 0.0) + 0.02) for a in ESSENTIAL_AA + ["glutamine"]) / (0.4 / 0.42)
        aa_s = min(aa_s, 1.0)
        o2 = med.get("O2", 0.2)
        hyp = max(0.0, (0.02 / (0.02 + o2) - 0.02 / 0.22) / (1 - 0.02 / 0.22))
        s["ampk"] = min(1.0, 3.0 * max(0.0, 1 - ec) + 1.5 * max(0.0, 1 - s["glyc"]))
        # unfolded-protein load: ER folding capacity, glycosylation, proteasome clearance, ER calcium store
        load = (max(0.0, 1.0 / max(F["er_folding"], 0.05) - 1.0) + 2.0 * self.env["er_stress"]
                + max(0.0, 1.0 - F["proteasome"]) + max(0.0, 1.0 - F["gpi_glycosylation"])
                + 1.5 * max(0.0, 1.0 - s["Ca_er"] / 400.0))
        s["U"] += dt_h * (0.5 * load - 0.5 * s["U"])
        perk = s["U"] / (s["U"] + 0.5)
        gcn2 = 1.0 - aa_s
        s["isr"] = float(np.clip(max(perk, gcn2), 0.0, 1.0))
        # redox
        s["ros"] = (0.5 + 0.5 * s["oxphos"]) / max(F["antioxidant"], 0.05) + self.env["ros"] + 0.5 * hyp
        ros_x = max(0.0, s["ros"] - 1.0)
        # DNA damage and repair
        repair = max(F["nhej"], 0.5 * F["hr"], 0.02)
        rs = self.env["replication_stress"] if self.phase == "S" else 0.0
        if self.phase == "S":
            rs = max(rs, 1.0 - min(s["dna_cap"], 1.0))
        s["D"] += dt_h * (0.35 * (1 + 5 * ros_x) + self.env["dsb_rate"] + 15.0 * rs * (self.phase == "S")
                          - 0.35 * repair * s["D"] / (1.0 + s["D"] / 40.0))     # repair saturates at high damage
        s["D"] = max(s["D"], 0.0)
        dmg = max(0.0, s["D"] - 1.0)
        atm = dmg / (dmg + 20.0) * min(rel("atm"), 1.0)
        atr = min(1.0, dmg / (dmg + 20.0) + rs) * min(rel("atr"), 1.0)
        self._p53_step(dt_h, atm, min(rel("mdm2"), 1.0), min(rel("tp53"), 2.0))
        p53_act = s["P53"] / self.p53_ref * (1.0 if not self.ko[self.sym["TP53"]] else 0.0)
        # ions and calcium
        self._ions(dt_h, F["na_k_atpase"], ec)
        ip3 = self.sig.activity("PLCG1") + 4.0 * self.env["gpcr"]
        self._calcium(dt_h, F["serca"], F["pmca"], ip3, ec)

        # ---------------- signalling (clamps from modules, then relax the network)
        A = A_REF
        self.sig.set(ATM=A + (1 - A) * atm, ATR=A + (1 - A) * atr, TP53=min(A * p53_act, 1.0),
                     PRKAA1=A + (1 - A) * s["ampk"], PRKAA2=A + (1 - A) * s["ampk"],
                     HIF1A=A + (1 - A) * hyp, EPAS1=A + (1 - A) * hyp,
                     EIF2AK3=A + (1 - A) * perk, EIF2AK4=A + (1 - A) * gcn2, ATF4=A + (1 - A) * s["isr"],
                     ATF6=A + (1 - A) * perk, XBP1=A + (1 - A) * perk, DDIT3=A + (1 - A) * s["isr"] ** 2,
                     NFE2L2=min(A * (1 + 2 * ros_x), 1.0), CALM1=min(A * s["Ca"] / 0.1, 1.0),
                     HSF1=A + (1 - A) * min(1.0, max(0.0, 1.0 - F["chaperones"]) + max(0.0, 1.0 - F["proteasome"])))
        e2f_lvl = {"G1": 0.5 + 2.0 * min(max(self.q, 0.0), 1.0), "S": 2.5, "G2": 1.0, "M": 0.5}[self.phase]
        g2m_lvl = {"G1": 0.6, "S": 1.0, "G2": 3.2, "M": 3.2}[self.phase]
        self.sig.set(E2F1=min(A * e2f_lvl, 1.0), E2F2=min(A * e2f_lvl, 1.0), E2F3=min(A * e2f_lvl, 1.0),
                     FOXM1=min(A * g2m_lvl, 1.0), MYBL2=min(A * g2m_lvl, 1.0))
        avail = np.ones(self.sig.n)
        gi = self.sig_gene[self.sig_has]
        lvl = np.where(self.p_ref[gi] > 0, functional[gi] / np.maximum(self.p_ref[gi] * size, 1e-12), 0.0)
        avail[self.sig_has] = np.where(lvl < 1.0, np.minimum(lvl / 0.7, 1.0), np.minimum(lvl, 4.0))   # reserve below, gain above
        self.sig.step(avail)
        erk = 0.5 * (self.sig.activity("MAPK1") + self.sig.activity("MAPK3"))
        akt = self.sig.activity("AKT1")
        s["mtor"] = float(np.clip(self.sig.activity("MTOR") * aa_s, 0.0, 1.5))
        s["autophagy"] = float(np.clip(1.0 - s["mtor"], 0.0, 1.0) + 0.5 * s["ampk"])

        # ---------------- gene expression (all genes)
        prot_rel = np.where(self.p_ref > 0, self.p / np.maximum(self.p_ref * size, 1e-12), 1.0)
        R = self.sig.regulation(prot_rel)
        g_tx = (min(F["rnap2"], F["splicing"], F["mrna_export"], F["cap_methylation"], F["chromatin"])
                * ec * (1 - self.env["transcription_block"]) * (0.1 if self.phase == "M" else 1.0))
        tx = self.k_tx * size * g_tx * R * self.tx_mult * (~self.ko) * dt_h
        dm = 1.0 - np.exp(-self.d_m * np.clip(F["mrna_decay"], 0.3, 1.5) * dt_h)
        g_tl = (min(F["ribosome"], F["translation_factors"], F["trna_charging"], F["chaperones"], F["rnap1"],
                    F["rnap3"], F["ribosome_biogenesis"], F["trna_processing"])
                * (0.4 + 0.6 * min(s["mtor"], 1.2)) * (1 - 0.7 * s["isr"]) * supply * ec
                * (1 - self.env["translation_block"]) * (0.3 if self.phase == "M" else 1.0))
        tl_gene = np.full(self.n, g_tl)
        for comp, machines in COMPARTMENT_MACHINE.items():
            tl_gene[self.comp_groups[comp]] *= min(1.0, min(F[x] for x in machines))
        tl_gene[self.is_mt] = F["mito_ribosome"] * ec          # mitochondrially encoded proteins
        tl = self.k_tl * self.m * tl_gene * dt_h
        g_deg = max(F["proteasome"], 0.05) * (1.0 + 0.5 * s["autophagy"] * F["v_atpase"])
        rate_p = self.d_p * g_deg
        # ubiquitin ligases with cell-cycle-gated activity: APC/C-Cdh1 (active in G1, silent S-M) and SCF (S phase)
        rate_p[self.gg["apc_deg"]] *= APC_ACTIVITY[self.phase] * F["apc"]
        rate_p[self.gg["scf_deg"]] *= SCF_ACTIVITY[self.phase]
        dp = 1.0 - np.exp(-rate_p * dt_h)
        if self.stochastic:
            self.m += self._poisson(tx) - self._binomial(self.m, dm)
            new_p = self._poisson(tl)
            lost_p = self._binomial(self.p, dp)
        else:
            self.m += tx - self.m * dm
            new_p, lost_p = tl, self.p * dp
        self.p += new_p - lost_p
        gain = float((new_p * self.mass).sum())
        loss = float((lost_p * self.mass).sum())
        s["mu"] = (gain - loss) / max(Mp, 1.0) / dt_h
        unthrottled = s["mu"] / max(supply, 0.05)
        s["mu_dem"] += 0.2 * (unthrottled - s["mu_dem"])
        s["turnover"] = loss / max(Mp, 1.0) / dt_h / 0.0185       # relative protein-turnover ATP cost

        # ---------------- cell cycle
        self._cell_cycle(dt_h, functional, size, F, erk, akt, atm, atr, dmg, rel)

        # ---------------- death decisions
        # BH3-only proteins and death-receptor signalling count once they clearly exceed their noisy baseline
        bh3 = (0.5 * max(0.0, rel("puma") - 2) + 0.5 * max(0.0, rel("noxa") - 2) + 0.5 * max(0.0, rel("bim") - 2))
        # death-inducing signalling complex: ligand x receptor x FADD x caspase-8, restrained by c-FLIP
        a = self.sig.a
        lig = lambda name: float(a[self.sig.pos[name]]) if name in self.sig.pos else 0.0
        disc = (lig("TNFSF10") * min(rel("dr"), 1.0) + lig("FASLG") * min(rel("fas"), 1.0)
                + 0.2 * lig("TNF") * min(rel("tnfr1"), 1.0))
        casp8 = 3.0 * disc * min(rel("fadd"), 1.0) * min(rel("casp8"), 1.0) / max(rel("cflar"), 0.3)
        stress = (0.2 * self.mitotic_h + 0.75 * max(0.0, rel("chop") - 3.0) + 1.0 * (1 - ec) + dmg / 200.0
                  + 0.5 * max(0.0, s["Ca"] - 1.0) + 0.5 * max(0.0, ros_x - 1.0))
        pro = min(rel("bax"), 2.0) * (1.0 + bh3 + casp8 + stress)
        anti = max(min(rel("anti"), 3.0) ** 0.8 * min(rel("xiap"), 3.0) ** 0.2 * (0.7 + 0.3 * min(akt, 2.0)), 0.02)
        ratio = pro / anti
        hill = ratio ** 4 / (39.0 + ratio ** 4)
        casp = min(F["caspases"], 1.0)
        n_sub = max(1, int(np.ceil(dt_h / 0.02)))
        for _ in range(n_sub):
            C = s["C"]
            s["C"] = float(np.clip(C + dt_h / n_sub * ((0.05 * hill + 2.5 * C * C / (0.09 + C * C)) * (1 - C) * casp - C), 0, 1))
        s["ratio"] = ratio
        s["low_energy_h"] = s["low_energy_h"] + dt_h if ec < 0.3 else 0.0
        if s["C"] > 0.5:
            self.death_cause = dict(pro=pro, anti=anti, bh3=bh3, casp8=casp8, stress=stress, mitotic_h=self.mitotic_h,
                                    bax=rel("bax"), anti_rel=rel("anti"), akt=akt, chop=rel("chop"), ec=ec, dsb=s["D"])
            self._die("apoptosis")
        elif s["low_energy_h"] > 3.0:
            self._die("necrosis (energy failure)")
        elif s["swell"] > 1.3:
            self._die("necrosis (osmotic lysis)")

        self.t += self.dt
        self._since_rec += self.dt
        if self._since_rec >= self.record_every:
            self._record(size, erk, akt)
            self._since_rec = 0.0

    # ------------------------------------------------------------------ metabolism coupling
    def _metabolism(self, functional, size, F):
        s = self.s
        level = dict(zip(self.met_genes, functional[self.met_gene_idx] / size))
        # iron-sulfur cluster and mitochondrial import machinery limit the enzymes that depend on them
        self.met.update_enzymes(level)
        s["atp_demand"] = 0.9 + 0.5 * s["pump"] + 0.6 * s.get("turnover", 1.0)
        res = self.met.solve(mu_demand=max(s["mu_dem"], 0.0) * s["supply"], atp_demand=s["atp_demand"])
        s["mu_cap"], s["atp_cap"] = res["mu_cap"], res["atp_cap"]
        s["EC"] = float(np.clip(res["atp_cap"] / s["atp_demand"], 0.0, 1.0)) ** 0.5 if res["atp_cap"] < s["atp_demand"] else 1.0
        want = max(s["mu_dem"], 1e-6)
        target = float(np.clip(res["mu_cap"] / want, 0.0, 1.0))
        s["supply"] += 0.7 * (target - s["supply"])
        self.exchange = res["exchange"]
        self.flux = res["v"]
        if res["v"] is not None:
            ref, ex = self.met.v_ref, self.met.exchange
            s["glyc"] = abs(res["v"][ex["glucose"]]) / abs(ref[ex["glucose"]])
            s["oxphos"] = abs(res["v"][ex["O2"]]) / max(abs(ref[ex["O2"]]), 1e-9)
        if self.phase == "S" or self._since_met == np.inf:
            col, coef = self.met.pool_cols["DNA"]
            lp = self.met.lp_pool
            lp.bounds(col, 0.0, 1000.0); lp.h.changeColCost(col, -1.0)
            obj, _ = lp.solve()
            lp.h.changeColCost(col, 0.0); lp.bounds(col, 0.0, 0.0)
            cap = -(obj or 0.0) / coef
            if not hasattr(self, "dna_cap_ref"):
                self.dna_cap_ref = max(cap, 1e-9)
            s["dna_cap"] = cap / self.dna_cap_ref          # dNTP synthesis capacity relative to the unperturbed cell
        self.met.step_medium(self.exchange, self.met_interval / 3600.0, self.culture_gdw_per_l * size)

    # ------------------------------------------------------------------ cell cycle
    def _cell_cycle(self, dt_h, functional, size, F, erk, akt, atm, atr, dmg, rel):
        s = self.s
        p21, p27 = rel("p21"), rel("p27")
        cki = 1.0 + 1.5 * max(p21 - 1.0, 0.0) + 1.0 * max(p27 - 1.0, 0.0)
        chk = max(atm, atr) * min(rel("chk1"), 1.0)
        if self.phase == "G1":
            rtk = np.mean([self.sig.activity(x) for x in ("EGFR", "IGF1R", "INSR", "FGFR1", "PDGFRB", "MET")
                           if x in self.sig.pos and self.sig.a_ref[self.sig.pos[x]] > 0] or [1.0])
            mitogen = float(np.clip(erk * akt, 0.0, 1.3) ** 3 * np.clip(self.sig.activity("MYC"), 0.0, 1.5)
                            * np.clip(rtk, 0.0, 1.3) ** 2)
            rb = min(rel("rb"), 2.0) * self.traits.get("rb_function", 1.0)
            rb_brake = (0.3 + 0.7 * rb) / (0.3 + 0.7 * self.traits.get("rb_function", 1.0))
            x = size / 0.95
            gate = min(2.0 * x ** 12 / (1.0 + x ** 12), 1.2)                 # cell-size checkpoint
            # cyclin D-CDK4/6 is needed only to inactivate RB: without functional RB the requirement vanishes
            cycd = 1.0 - min(rb, 1.0) * (1.0 - min(_sat(min(rel("cycD"), rel("cdk46"))), 1.0)) \
                + 0.5 * max(min(rel("cycD"), rel("cdk46"), 3.0) - 1.0, 0.0)
            drive = (mitogen * cycd / cki / rb_brake * gate
                     * (1 - 0.8 * atm * min(rel("chk2"), 1.0)) * _sat(min(rel("cycE"), rel("cdk2"))) ** 0.5
                     * _sat(rel("e2f")) * min(s["supply"], 1.0) * s["EC"])
            if not self.arrested:
                self.q += dt_h * max(drive - 0.4, 0.0) / 0.6
            s["g1_drive"] = drive
            if self.q >= 1.0 and F["licensing"] > 0.3:
                self.phase, self.r = "S", 0.0
        elif self.phase == "S":
            rate = (F["replication"] * min(s["dna_cap"], 1.0) * (1 - 0.95 * chk) * F["chromatin"]
                    * (1 - 0.9 * self.env["replication_stress"]) * min(_sat(min(rel("cycA"), rel("cdk2"))), 1.0) * s["EC"])
            self.r += dt_h * rate / 8.0
            if self.r >= 1.0:
                self.r, self.phase, self.b = 1.0, "G2", 0.0
        elif self.phase == "G2":
            drive = (_sat(min(rel("cycB"), rel("cdk1"))) * _sat(rel("cdc25")) / max(rel("wee1"), 0.2) ** 0.5
                     / (1.0 + 0.5 * max(p21 - 1.0, 0.0)) * (1 - 0.97 * chk) * s["EC"])
            self.b += dt_h * max(drive - 0.4, 0.0) / 1.8
            if self.b >= 1.0:
                self.phase, self.mprog, self.mitotic_h = "M", 0.0, 0.0
                if dmg > 60.0:
                    self._die("mitotic catastrophe")
        elif self.phase == "M":
            rate = (min(F["spindle"], F["apc"], F["cohesin_condensin"])
                    * (1 - self.env["spindle_poison"]) * s["EC"])
            self.mprog += dt_h * rate
            self.mitotic_h += dt_h
            if self.mprog >= 1.0:
                if F["cytokinesis"] > 0.3:
                    self._divide()
                else:                                   # cytokinesis failure: tetraploid G1 arrest
                    self.phase, self.q, self.mitotic_h, self.arrested = "G1", 0.0, 0.0, True
                    self.fate = "cytokinesis failure (tetraploid arrest)"

    def _degrade(self, group, keep):
        idx = self.gg[group]
        self.p[idx] = np.round(self.p[idx] * keep) if self.stochastic else self.p[idx] * keep

    def _divide(self):
        self._degrade("apc_deg", 0.1)
        if self.stochastic:
            self.m = self.rng.binomial(self.m.astype(np.int64), 0.5).astype(float)
            self.p = self.rng.binomial(self.p.astype(np.int64), 0.5).astype(float)
        else:
            self.m, self.p = self.m * 0.5, self.p * 0.5
        self.s["D"] *= 0.5
        self.phase, self.q, self.r, self.b, self.mprog, self.mitotic_h = "G1", 0.0, 0.0, 0.0, 0.0, 0.0
        self.generation += 1
        self.divisions.append(self.t / 3600.0)

    def _die(self, how):
        self.alive, self.fate = False, how
        self.death_time = self.t / 3600.0

    # ------------------------------------------------------------------ bookkeeping
    def _record(self, size, erk, akt):
        h, s = self.history, self.s
        ex = self.exchange
        vals = dict(t=self.t / 3600.0, phase=self.phase, size=size, mu=s["mu"], mu_cap=s["mu_cap"], EC=s["EC"],
                    supply=s["supply"], P53=s["P53"] / self.p53_ref, D=s["D"], C=s["C"], U=s["U"], Vm=s["Vm"],
                    Ca=s["Ca"], swell=s["swell"], mtor=s["mtor"], isr=s["isr"], ampk=s["ampk"], ros=s["ros"],
                    r=self.r, mrna_total=float(self.m.sum()), protein_total=float(self.p.sum()),
                    glucose=self.met.medium.get("glucose", 0.0), lactate=self.met.medium.get("L-lactate", 0.0),
                    q_glc=ex.get("glucose", 0.0), q_lac=ex.get("L-lactate", 0.0), q_o2=ex.get("O2", 0.0),
                    q_gln=ex.get("glutamine", 0.0), ERK=erk, AKT=akt, generation=self.generation)
        for k, v in vals.items():
            h[k].append(v)
        for w in self.watch:
            i = self.sym[w]
            h[f"p:{w}"].append(self.p[i])
            h[f"m:{w}"].append(self.m[i])

    def run(self, hours: float, until_division: bool = False):
        """Advance the cell. Returns self so calls can be chained."""
        n = int(round(hours * 3600.0 / self.dt))
        gen = self.generation
        for _ in range(n):
            if not self.alive or (until_division and self.generation > gen):
                break
            self.step()
        return self

    def frame(self):
        import pandas as pd
        return pd.DataFrame(self.history)

    def summary(self) -> dict:
        t_h = self.t / 3600.0
        size = float((self.p * self.mass).sum()) / self.Mp_ref
        return dict(alive=self.alive, fate=self.fate, hours=t_h, divisions=self.generation, phase=self.phase,
                    size=size, growth=np.log2(max(size, 1e-9) / LN2 * 2.0 ** self.generation) / max(t_h, 1e-9) * LN2,
                    mu_cap=self.s["mu_cap"], EC=self.s["EC"], p53=self.s["P53"] / self.p53_ref, dsb=self.s["D"])

    def reset_shared(self):
        """Return the shared metabolic / signalling models to the unperturbed reference environment."""
        from .metabolism import MEDIUM
        self.sig.a = self.sig.a_ref.copy()
        self.sig.clamp = {}
        self.met.set_medium(**{k: v[0] for k, v in MEDIUM.items()}, **{"L-lactate": 0.0})
        self._since_met = np.inf
        return self

    def clone(self, seed=None):
        """Independent copy of the molecular state that shares the (expensive) metabolic and signalling models."""
        met, sig = self.met, self.sig
        self.met = self.sig = None
        new = copy.deepcopy(self)
        self.met, self.sig = met, sig
        new.met, new.sig = met, sig
        if seed is not None:
            new.rng = np.random.default_rng(seed)
        return new
