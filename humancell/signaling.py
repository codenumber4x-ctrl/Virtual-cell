"""Genome-scale signalling and transcriptional regulation.

Topology is literature-curated (OmniPath signed/directed interactions, CollecTRI regulons).
Quantitative interaction strengths have not been measured genome-wide, so the dynamics are a
signed-network propagation model linearised around the reference (growing, serum-fed) state:

    a_i = avail_i * sigmoid( logit(a_ref) + beta * sum_j W_ij (a_j - a_j_ref) )
    W_ij = sign_ij * c_ij / sum_j c_ij

where c_ij is the literature support of the edge (log of its curation effort), and edges of the
textbook backbone of each pathway (`CORE`, hand-curated as in every whole-cell model) dominate the
input of their target. `beta` is set from the leading eigenvalue of W so that the reference state
is a stable fixed point. Nodes can be clamped by the environment (ligands), by the kinetic
modules (ATM, AMPK, p53, E2F ...) or scaled by protein availability (knock-outs, drugs).
This layer is *logic-level*: it gets directions and relative magnitudes of pathway responses,
not absolute kinetics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from . import data

A_REF = 0.25
EPS = 0.02
LIGANDS = ["EGF", "IGF1", "INS", "FGF2", "PDGFB", "HGF", "TGFB1", "TNF", "IL6", "IFNG", "IL1B", "WNT3A",
           "TNFSF10", "FASLG", "VEGFA", "NRG1", "IFNB1", "CXCL12"]
SERUM = ["EGF", "IGF1", "INS", "FGF2", "PDGFB", "HGF"]          # mitogens present in the reference medium
TARGET_GAIN = 0.8     # log2 fold change of a target gene per unit of summed, normalised TF log2 activity
CORE_SHARE = 0.7      # fraction of a node's input carried by its textbook-backbone regulators

_CORE = """
EGF>EGFR IGF1>IGF1R INS>INSR FGF2>FGFR1 PDGFB>PDGFRB HGF>MET NRG1>ERBB3 VEGFA>KDR
EGFR>GRB2 IGF1R>IRS1 INSR>IRS1 FGFR1>FRS2 FRS2>GRB2 PDGFRB>GRB2 MET>GRB2 GRB2>SOS1 IRS1>GRB2
SOS1>KRAS SOS1>HRAS SOS1>NRAS KRAS>RAF1 KRAS>BRAF HRAS>RAF1 NRAS>RAF1 RAF1>MAP2K1 RAF1>MAP2K2 BRAF>MAP2K1 BRAF>MAP2K2
MAP2K1>MAPK1 MAP2K1>MAPK3 MAP2K2>MAPK1 MAP2K2>MAPK3 MAPK1>ELK1 MAPK3>ELK1 MAPK1>MYC MAPK3>MYC MAPK1>FOS MAPK3>FOS
MAPK1>RPS6KA1 MAPK3>RPS6KA1 MAPK1>ETS1 MAPK1>JUN ELK1>FOS
IRS1>PIK3CA EGFR>PIK3CA PDGFRB>PIK3CA KRAS>PIK3CA PIK3CA>PDPK1 PTEN-|PDPK1 PTEN-|AKT1 PDPK1>AKT1 PDPK1>AKT2 MTOR>AKT1
AKT1-|FOXO3 AKT1-|FOXO1 AKT1-|GSK3B AKT1-|TSC2 AKT1>MDM2 AKT1-|BAD AKT1-|CDKN1B TSC2-|RHEB RHEB>MTOR MTOR>RPS6KB1 MTOR-|EIF4EBP1
PRKAA1>TSC2 PRKAA1-|MTOR PRKAA1-|RPTOR STK11>PRKAA1 PRKAA1>ULK1 MTOR-|ULK1 PRKAA1>PPARGC1A PRKAA1-|ACACA
GSK3B-|CTNNB1 GSK3B-|MYC GSK3B-|CCND1 WNT3A>FZD1 FZD1>DVL1 DVL1-|GSK3B CTNNB1>TCF7L2 CTNNB1>LEF1 APC-|CTNNB1 AXIN1-|CTNNB1
TNF>TNFRSF1A TNFRSF1A>TRADD TRADD>TRAF2 TRADD>FADD TRAF2>MAP3K7 IL1B>IL1R1 IL1R1>MYD88 MYD88>IRAK1 IRAK1>TRAF6 TRAF6>MAP3K7
MAP3K7>IKBKB MAP3K7>MAP2K4 MAP3K7>MAP2K3 IKBKB-|NFKBIA NFKBIA-|RELA NFKBIA-|NFKB1 IKBKB>RELA MAP2K4>MAPK8 MAP2K4>MAPK9 MAPK8>JUN MAP2K3>MAPK14 MAPK14>ATF2
TNFSF10>TNFRSF10B TNFRSF10B>FADD FASLG>FAS FAS>FADD FADD>CASP8 CFLAR-|CASP8 CASP8>CASP3 CASP8>BID BID>BAX
BAX>CYCS BAK1>CYCS BCL2-|BAX BCL2L1-|BAX MCL1-|BAK1 CYCS>APAF1 APAF1>CASP9 CASP9>CASP3 XIAP-|CASP3 XIAP-|CASP9 DIABLO-|XIAP
TGFB1>TGFBR2 TGFBR2>TGFBR1 TGFBR1>SMAD2 TGFBR1>SMAD3 SMAD7-|TGFBR1 SMAD3>SMAD4 SMAD2>SMAD4
IL6>IL6R IL6R>IL6ST IL6ST>JAK1 IL6ST>JAK2 JAK1>STAT3 JAK2>STAT3 SOCS3-|JAK2 SOCS3-|JAK1
IFNG>IFNGR1 IFNGR1>JAK1 IFNGR1>JAK2 JAK1>STAT1 JAK2>STAT1 IFNB1>IFNAR1 IFNAR1>TYK2 TYK2>STAT1 TYK2>STAT2 STAT1>IRF1
ATM>CHEK2 ATM>TP53 CHEK2>TP53 ATR>CHEK1 CHEK1-|CDC25A CHEK1-|CDC25C CHEK2-|CDC25A MDM2-|TP53 TP53>CDKN1A ATM>H2AX
CDKN1A-|CDK2 CDKN1A-|CDK4 CDKN1B-|CDK2 CCND1>CDK4 CCND1>CDK6 CDK4-|RB1 CDK6-|RB1 CDK2-|RB1 RB1-|E2F1 CCNE1>CDK2 CDC25A>CDK2
E2F1>CCNE1 E2F1>CCNA2 CCNA2>CDK2 CCNB1>CDK1 CDC25C>CDK1 WEE1-|CDK1 CDK1>FOXM1 PLK1>FOXM1 PLK1>CDC25C MYC>CCND1 MYC>E2F1
HIF1A>EPAS1 VHL-|HIF1A EGLN1-|HIF1A EIF2AK3>EIF2S1 EIF2AK4>EIF2S1 EIF2AK3>NFE2L2 KEAP1-|NFE2L2
CXCL12>CXCR4 CALM1>PPP3CA PPP3CA>NFATC1 CALM1>CAMK2A PLCG1>PRKCA EGFR>PLCG1 PRKCA>RAF1 SRC>STAT3 EGFR>SRC SRC>PTK2
YAP1>TEAD1 LATS1-|YAP1 STK3>LATS1 NF2>LATS1 NOTCH1>RBPJ JAG1>NOTCH1 SHH>PTCH1 PTCH1-|SMO SMO>GLI1 SUFU-|GLI1
"""
CORE = []
for _tok in _CORE.split():
    if "-|" in _tok:
        _s, _t = _tok.split("-|"); CORE.append((_s, _t, -1))
    else:
        _s, _t = _tok.split(">"); CORE.append((_s, _t, 1))


class Signaling:
    def __init__(self, genes: pd.DataFrame, contraction: float = 0.9):
        sym2idx = {s: i for i, s in enumerate(genes["symbol"])}
        self.genes = genes
        edges = data.signaling_edges()
        core = pd.DataFrame(CORE, columns=["source", "target", "sign"])
        core = core[core["source"].isin(sym2idx) | core["source"].isin(LIGANDS)]
        core = core[core["target"].isin(sym2idx)]
        core["effort"], core["core"] = 0, True
        edges["core"] = False
        edges = pd.concat([core, edges]).drop_duplicates(["source", "target"]).reset_index(drop=True)
        nodes = sorted(set(edges["source"]) | set(edges["target"]))
        self.nodes = nodes
        self.pos = {s: i for i, s in enumerate(nodes)}
        n = self.n = len(nodes)
        self.gene_idx = np.array([sym2idx.get(s, -1) for s in nodes])       # row in gene table, -1 if absent
        src = edges["source"].map(self.pos).values
        tgt = edges["target"].map(self.pos).values
        is_core = edges["core"].values
        c = np.log2(1.0 + edges["effort"].values.astype(float))
        tot_db = np.bincount(tgt, weights=np.where(is_core, 0.0, c), minlength=n)
        n_core = np.bincount(tgt, weights=is_core.astype(float), minlength=n)
        share_core = np.where(n_core > 0, np.where(tot_db > 0, CORE_SHARE, 1.0), 0.0)
        w = np.where(is_core, share_core[tgt] / np.maximum(n_core[tgt], 1),
                     (1 - share_core[tgt]) * c / np.maximum(tot_db[tgt], 1e-12))
        edges["weight"] = w
        self.W = sp.csr_matrix((edges["sign"].values * w, (tgt, src)), shape=(n, n))
        self.indeg = np.bincount(tgt, minlength=n).astype(float)
        lam = eigs(self.W, k=6, which="LR", return_eigenvectors=False, maxiter=5000, tol=1e-4)
        self.rho = float(max(lam.real.max(), 0.2))
        self.beta = contraction / (self.rho * A_REF * (1 - A_REF))
        self.edges = edges

        # nodes whose protein is not expressed in this cell line are silent
        prot = genes["protein"].values
        self.expressed = np.array([(gi >= 0 and prot[gi] > 0) for gi in self.gene_idx])
        self.is_ligand = np.array([s in LIGANDS for s in nodes])
        self.a_ref = np.where(self.expressed | self.is_ligand, A_REF, 0.0)
        for lig in LIGANDS:                       # ligands are environmental: only serum mitogens are present
            if lig in self.pos:
                self.a_ref[self.pos[lig]] = A_REF if lig in SERUM else 0.0
        # the reference state is defined as the fixed point: u = 0 for every node
        self.logit_ref = np.log(A_REF / (1 - A_REF))
        self.a = self.a_ref.copy()
        self.clamp = {}                          # node index -> activity

        # ---- transcriptional regulation (TF -> target gene)
        reg = data.regulon_edges()
        reg = reg[reg["target"].isin(sym2idx)]
        self.tfs = sorted(set(reg["tf"]))
        self.tf_pos = {t: i for i, t in enumerate(self.tfs)}
        tg = reg["target"].map(sym2idx).values
        tf = reg["tf"].map(self.tf_pos).values
        kg = np.bincount(tg, minlength=len(genes)).astype(float)
        self.R = sp.csr_matrix((reg["sign"].values / np.sqrt(kg[tg]), (tg, tf)), shape=(len(genes), len(self.tfs)))
        self.tf_node = np.array([self.pos.get(t, -1) for t in self.tfs])     # signalling node of each TF
        self.tf_gene = np.array([sym2idx.get(t, -1) for t in self.tfs])      # gene row of each TF
        self.reg = reg
        # Transcription factors regulate each other's genes; the reference state is only a stable fixed point
        # of that loop if the loop gain is below one. The regulatory gain is set from the leading eigenvalue
        # of the TF -> TF-gene sub-network (same principle as `beta` above).
        ok = self.tf_gene >= 0
        T = self.R[self.tf_gene[ok]][:, np.where(ok)[0]]
        lam = eigs(T.asfptype(), k=6, which="LR", return_eigenvectors=False, maxiter=5000, tol=1e-4)
        self.tf_loop = float(lam.real.max())
        self.gain = float(min(1.0, 0.6 / max(self.tf_loop, 1e-6)))
        # Only the TF genes close that loop; every other target gene responds with a larger gain.
        self.gene_gain = np.full(len(genes), TARGET_GAIN)
        self.gene_gain[self.tf_gene[ok]] = self.gain

    # ------------------------------------------------------------------
    def set(self, **activity):
        """Clamp nodes, e.g. set(EGF=1.0, TNF=0.5); value None releases the clamp."""
        for name, val in activity.items():
            if name in self.pos:
                if val is None:
                    self.clamp.pop(self.pos[name], None)
                else:
                    self.clamp[self.pos[name]] = float(val)

    def serum(self, level: float):
        """Scale all serum mitogens (1 = reference medium, 0 = serum-free)."""
        self.set(**{lig: A_REF * level for lig in SERUM})

    def step(self, avail: np.ndarray, n_iter: int = 3, relax: float = 0.6):
        """Relax towards the fixed point. `avail` = functional protein level relative to reference per node."""
        a = self.a
        ci = np.fromiter(self.clamp.keys(), dtype=int, count=len(self.clamp))
        cv = np.fromiter(self.clamp.values(), dtype=float, count=len(self.clamp))
        for _ in range(n_iter):
            u = self.W @ (a - self.a_ref)
            target = avail / (1.0 + np.exp(-np.clip(self.logit_ref + self.beta * u, -50.0, 50.0)))
            target = np.where(self.is_ligand, self.a_ref, np.clip(target, 0.0, 1.0))
            target[~(self.expressed | self.is_ligand)] = 0.0
            a = a + relax * (target - a)
            if len(ci):
                a[ci] = cv
        self.a = a
        return a

    def activity(self, name: str) -> float:
        """Activity relative to the reference state (1 = reference)."""
        i = self.pos.get(name)
        return 1.0 if i is None or self.a_ref[i] == 0 else float(self.a[i] / self.a_ref[i])

    def tf_log2(self, protein_rel: np.ndarray) -> np.ndarray:
        """log2 fold change of each TF's activity: signalling state if it is a signalling node, else its abundance."""
        out = np.zeros(len(self.tfs))
        has = self.tf_node >= 0
        node = self.tf_node[has]
        out[has] = np.where(self.a_ref[node] > 0, np.log2((np.maximum(self.a[node], 0.0) + EPS) / (self.a_ref[node] + EPS)), 0.0)
        g = self.tf_gene
        only_abund = (~has) & (g >= 0)
        out[only_abund] = np.log2(np.clip(protein_rel[g[only_abund]], 0.25, 4.0))
        return out

    def regulation(self, protein_rel: np.ndarray, gain: float | None = None) -> np.ndarray:
        """Per-gene multiplicative change of the transcription rate."""
        gain = self.gene_gain if gain is None else gain
        return np.clip(2.0 ** (gain * (self.R @ self.tf_log2(protein_rel))), 0.05, 20.0)
