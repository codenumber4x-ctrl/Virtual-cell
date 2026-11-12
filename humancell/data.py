"""Data layer: download, cache and harmonise the public resources the virtual cell is built from.

Every number that enters the model comes through this module, so each one can be traced to
a source (measured) or to an explicit fallback rule (inferred). `build_gene_table` records
that provenance per gene in the `*_src` columns.
"""
from __future__ import annotations

import gzip
import io
import os
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("HUMANCELL_DATA", ROOT / "data"))

PAX = "https://pax-db.org/downloads/latest/datasets/9606/"
SOURCES = {
    "Human-GEM.yml": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/Human-GEM.yml",
    "genes.tsv": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/genes.tsv",
    "rna_celline.tsv.zip": "https://www.proteinatlas.org/download/tsv/rna_celline.tsv.zip",
    "subcellular_location.tsv.zip": "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip",
    "omnipath_sig.tsv": "https://omnipathdb.org/interactions?datasets=omnipath&genesymbols=1&fields=sources,references,curation_effort&organisms=9606",
    "collectri.tsv": "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=1&organisms=9606",
    "complexes.tsv": "https://omnipathdb.org/complexes",
    "uniprot.tsv": "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true+AND+organism_id:9606&format=tsv&fields=accession,gene_primary,length,mass,cc_subcellular_location",
    "CEGv2.txt": "https://raw.githubusercontent.com/hart-lab/bagel/master/CEGv2.txt",
    "NEGv1.txt": "https://raw.githubusercontent.com/hart-lab/bagel/master/NEGv1.txt",
    "paxdb_human.txt": PAX + "9606-WHOLE_ORGANISM-integrated.txt",
    "px_hela_mueller.txt": PAX + "9606-PXD014877_Mueller_Nature_2020_Homo_sapiens_HeLa.txt",
    "schwan.xls": "https://static-content.springer.com/esm/art%3A10.1038%2Fnature10098/MediaObjects/41586_2011_BFnature10098_MOESM304_ESM.xls",
}

# HPA cell-line names for the lines kept in the compact expression subset
CELL_LINES = ["HeLa", "Hep-G2", "K-562", "U2OS", "A-549", "MCF-7", "HEK293", "hTERT-RPE1"]

# Cell-line specific proteomes available in PaxDB (others fall back to the integrated human set)
CELL_PROTEOME = {"HeLa": "px_hela_mueller.txt"}


def fetch(name: str) -> Path:
    """Return the local path of a cached source file, downloading it if needed."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    if not path.exists() or path.stat().st_size == 0:
        subprocess.run(["curl", "-sSL", "--retry", "4", "-m", "900", "-o", str(path), SOURCES[name]], check=True)
    return path


def fetch_all() -> None:
    for name in SOURCES:
        if name != "rna_celline.tsv.zip":
            fetch(name)
    expression_table()


def expression_table() -> pd.DataFrame:
    """HPA RNA-seq (nTPM) for the supported cell lines, genes x cell lines."""
    sub = DATA / "rna_celline_subset.tsv"
    if not sub.exists():
        with zipfile.ZipFile(fetch("rna_celline.tsv.zip")) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), sep="\t")
        df[df["Cell line"].isin(CELL_LINES)].to_csv(sub, sep="\t", index=False)
    df = pd.read_csv(sub, sep="\t")
    return df.pivot_table(index=["Gene", "Gene name"], columns="Cell line", values="nTPM").reset_index()


# ----------------------------------------------------------------------------------------
# Compartments
# ----------------------------------------------------------------------------------------
HPA_COMPARTMENT = {
    "nucleus": ["Nucleoplasm", "Nuclear speckles", "Nuclear bodies", "Nucleoli", "Nucleoli fibrillar center",
                "Nucleoli rim", "Nuclear membrane", "Kinetochore", "Mitotic chromosome"],
    "cytosol": ["Cytosol", "Cytoplasmic bodies", "Rods & Rings", "Aggresome"],
    "mitochondria": ["Mitochondria"],
    "er": ["Endoplasmic reticulum"],
    "golgi": ["Golgi apparatus"],
    "plasma_membrane": ["Plasma membrane", "Cell Junctions", "Focal adhesion sites"],
    "cytoskeleton": ["Actin filaments", "Microtubules", "Intermediate filaments", "Centrosome", "Centriolar satellite",
                     "Microtubule ends", "Mitotic spindle", "Cytokinetic bridge", "Midbody", "Midbody ring",
                     "Cleavage furrow", "Basal body", "Primary cilium", "Primary cilium tip",
                     "Primary cilium transition zone"],
    "vesicles": ["Vesicles", "Lysosomes", "Endosomes", "Peroxisomes", "Lipid droplets"],
}
_HPA2C = {loc: c for c, locs in HPA_COMPARTMENT.items() for loc in locs}
UNIPROT_KEYS = [("mitochondri", "mitochondria"), ("endoplasmic reticulum", "er"), ("golgi", "golgi"),
                ("secreted", "secreted"), ("nucleus", "nucleus"), ("cell membrane", "plasma_membrane"),
                ("lysosom", "vesicles"), ("peroxisom", "vesicles"), ("endosom", "vesicles"),
                ("cytoskeleton", "cytoskeleton"), ("cytoplasm", "cytosol"), ("membrane", "plasma_membrane")]
COMPARTMENTS = list(HPA_COMPARTMENT) + ["secreted"]


def _uniprot_compartment(text) -> str | None:
    if not isinstance(text, str):
        return None
    t = text.lower()
    first, best = len(t) + 1, None
    for key, comp in UNIPROT_KEYS:
        i = t.find(key)
        if 0 <= i < first:
            first, best = i, comp
    return best


# ----------------------------------------------------------------------------------------
# Gene table
# ----------------------------------------------------------------------------------------
# Physical reference values for a HeLa cell (BioNumbers / Milo & Phillips, Cell Biology by the Numbers)
PHYS = dict(
    volume_fl=2600.0,          # cell volume, fL (BNID 103725: ~2600-3000 um^3)
    dry_mass_pg=450.0,         # dry mass per cell
    protein_frac=0.60,         # protein fraction of dry mass
    mrna_total=4.0e5,          # mRNA molecules per cell
    doubling_h=22.0,           # doubling time in rich medium
    aa_mass_da=110.0,          # mean residue mass
    genome_bp=6.4e9,           # diploid genome (HeLa is hypotriploid; diploid reference used)
)
TPM_MIN = 1.0
K_TL_MAX = 1000.0              # proteins per mRNA per hour
DEFAULT_T_HALF_MRNA = 9.0      # h, median of Schwanhaeusser et al. 2011
DEFAULT_T_HALF_PROT = 46.0     # h, median of Schwanhaeusser et al. 2011


def _paxdb(name: str) -> pd.Series:
    df = pd.read_csv(fetch(name), sep="\t", comment="#", header=None, usecols=[0, 1, 2], names=["symbol", "sid", "ppm"])
    return df.groupby("symbol")["ppm"].sum()


def _halflives() -> pd.DataFrame:
    """Protein/mRNA half-lives measured in mouse fibroblasts, mapped to human by gene symbol."""
    x = pd.read_excel(fetch("schwan.xls"), sheet_name=0)
    out = pd.DataFrame({
        "symbol": x["Gene Names"].astype(str).str.split(";").str[0].str.upper(),
        "t_half_prot": pd.to_numeric(x["Protein half-life average [h]"], errors="coerce"),
        "t_half_mrna": pd.to_numeric(x["mRNA half-life average [h]"], errors="coerce"),
    })
    return out.groupby("symbol").median()


def build_gene_table(cell_line: str = "HeLa", refresh: bool = False) -> pd.DataFrame:
    """One row per protein-coding gene with every per-gene parameter of the model.

    Abundances are balanced-growth, cell-cycle-averaged copies per cell; rate constants are
    derived from them so that the reference state is a steady state of the model:
        k_tx = m (d_m + mu)          transcripts / h            (per cell, all gene copies)
        k_tl = p (d_p + mu) / m      proteins / mRNA / h
    """
    cache = DATA / f"gene_table_{cell_line}.csv.gz"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0)

    expr = expression_table()
    g = pd.DataFrame({"ensembl": expr["Gene"], "symbol": expr["Gene name"], "tpm": expr[cell_line]})
    g = g.drop_duplicates("ensembl").set_index("ensembl")

    up = pd.read_csv(fetch("uniprot.tsv"), sep="\t")
    up.columns = ["uniprot", "symbol", "length", "mass", "cc"]
    up = up.dropna(subset=["symbol"]).drop_duplicates("symbol").set_index("symbol")
    g = g.join(up, on="symbol")
    g["length_src"] = np.where(g["length"].notna(), "UniProt", "default")
    g["length"] = g["length"].fillna(up["length"].median())
    g["mass"] = g["mass"].fillna(g["length"] * PHYS["aa_mass_da"])

    # --- compartment: HPA imaging first, UniProt annotation second
    with zipfile.ZipFile(fetch("subcellular_location.tsv.zip")) as z:
        loc = pd.read_csv(z.open(z.namelist()[0]), sep="\t").set_index("Gene")
    main = loc["Main location"].reindex(g.index)
    hpa_c = main.map(lambda s: _HPA2C.get(s.split(";")[0]) if isinstance(s, str) else None)
    sec = loc["Extracellular location"].reindex(g.index).notna() & hpa_c.isna()
    up_c = g["cc"].map(_uniprot_compartment)
    g["compartment"] = hpa_c.where(hpa_c.notna(), up_c)
    g.loc[sec, "compartment"] = "secreted"
    g["compartment_src"] = np.where(hpa_c.notna(), "HPA", np.where(up_c.notna(), "UniProt", "default"))
    g["compartment"] = g["compartment"].fillna("cytosol")
    g.loc[g["symbol"].str.startswith("MT-"), "compartment"] = "mitochondria"
    g = g.drop(columns="cc")

    # --- mRNA copies per cell (HPA RNA-seq). A few genes that the cell-line proteome detects
    # robustly have ~0 TPM in HPA (annotation artefacts, e.g. POLR2A); their mRNA is imputed.
    tpm = g["tpm"].fillna(0.0).copy()
    cl_ppm = g["symbol"].map(_paxdb(CELL_PROTEOME[cell_line])) if cell_line in CELL_PROTEOME else pd.Series(np.nan, index=g.index)
    g["mrna_src"] = "HPA RNA-seq"
    ok = (tpm >= 1) & (cl_ppm > 0)
    if ok.sum() > 100:
        b, a = np.polyfit(np.log10(cl_ppm[ok]), np.log10(tpm[ok]), 1)
        fix = (tpm < 1) & (cl_ppm > 0)
        tpm[fix] = 10 ** (a + b * np.log10(cl_ppm[fix]))
        g.loc[fix, "mrna_src"] = "imputed from proteome"
    # Standard detection threshold for context-specific models: below 1 TPM (~0.4 mRNA copies per cell) a
    # transcript is indistinguishable from noise and the gene is treated as not expressed in this cell line.
    g.loc[tpm < TPM_MIN, "mrna_src"] = "below detection (<1 TPM)"
    tpm[tpm < TPM_MIN] = 0.0
    g["mrna"] = tpm / tpm.sum() * PHYS["mrna_total"]

    # --- protein copies per cell: cell-line proteome > integrated human proteome > mRNA regression
    prot_total_da = PHYS["dry_mass_pg"] * PHYS["protein_frac"] * 1e-12 * 6.022e23
    ppm = pd.Series(np.nan, index=g.index)
    src = pd.Series("mRNA-regression", index=g.index)
    for s, label in [(cl_ppm, f"PaxDB {cell_line}"), (g["symbol"].map(_paxdb("paxdb_human.txt")), "PaxDB integrated")]:
        take = ppm.isna() & s.notna() & (s > 0) & (g["mrna"] > 0)
        if label == "PaxDB integrated":
            # the tissue-integrated set is only trusted where the gene is transcribed in this line,
            # and is rescaled to the cell-line proteome on the overlap
            both = ppm.notna() & s.notna() & (s > 0)
            if both.sum() > 100:
                s = s * np.exp(np.median(np.log(ppm[both]) - np.log(s[both])))
            take &= tpm >= 1.0
        ppm[take] = s[take]
        src[take] = label
    known = ppm.notna() & (g["mrna"] > 0)
    slope, icpt = np.polyfit(np.log10(g.loc[known, "mrna"]), np.log10(ppm[known]), 1)
    pred = 10 ** (icpt + slope * np.log10(g["mrna"].clip(lower=1e-3)))
    ppm = ppm.where(ppm.notna(), pred.where(g["mrna"] > 0, 0.0))
    # proteins/mRNA ratios beyond what translation can sustain (tissue-secreted proteins in the
    # integrated set) are capped at the 99th percentile of the cell-line-measured ratios
    ratio = ppm / g["mrna"].clip(lower=1e-9)
    cap = ratio[src == src.iloc[np.argmax((src != "mRNA-regression").values)]].quantile(0.99) if known.any() else np.inf
    capped = (ratio > cap) & (src != f"PaxDB {cell_line}")
    ppm[capped] = cap * g["mrna"][capped]
    src[capped] = src[capped] + " (capped)"
    weight = ppm * g["mass"]
    g["protein"] = ppm * prot_total_da / weight.sum()
    g["protein_src"] = src.where(g["protein"] > 0, "not expressed")

    # --- half-lives
    hl = _halflives()
    g = g.join(hl, on="symbol")
    g["t_half_prot_src"] = np.where(g["t_half_prot"].notna(), "Schwanhaeusser 2011 (mouse ortholog)", "default median")
    g["t_half_mrna_src"] = np.where(g["t_half_mrna"].notna(), "Schwanhaeusser 2011 (mouse ortholog)", "default median")
    g["t_half_prot"] = g["t_half_prot"].fillna(DEFAULT_T_HALF_PROT).clip(0.25, 500)
    g["t_half_mrna"] = g["t_half_mrna"].fillna(DEFAULT_T_HALF_MRNA).clip(0.25, 100)
    # Short-lived regulators that the mouse data set does not resolve (literature values, h)
    fast = {"MYC": (0.5, 0.5), "MCL1": (0.6, 1.5), "TP53": (0.4, 3.0), "MDM2": (0.5, 1.0), "CCND1": (0.5, 1.5),
            "CCNE1": (0.5, 2.0), "CCNB1": (1.0, 4.0), "CCNA2": (1.0, 4.0), "CDKN1A": (0.5, 2.0), "FOS": (0.5, 0.3),
            "JUN": (1.5, 0.5), "HIF1A": (0.1, 2.0), "ATF4": (0.5, 2.0), "CDC25A": (0.4, 2.0), "ODC1": (0.3, 2.0),
            "NFKBIA": (0.7, 0.5), "CDT1": (0.5, 2.0), "E2F1": (1.5, 2.0), "BCL2L11": (3.0, 3.0), "PMAIP1": (0.7, 1.0),
            "CDKN1B": (2.0, 3.0), "PLK1": (3.0, 4.0), "AURKA": (2.0, 4.0), "CDC20": (2.0, 4.0), "PTTG1": (1.0, 4.0),
            "CCNB2": (1.0, 4.0), "CDC6": (1.0, 2.0), "CCNE2": (0.5, 2.0), "FOXM1": (2.0, 3.0), "MYBL2": (2.0, 3.0), "CTNNB1": (1.0, 5.0), "NFE2L2": (0.3, 2.0), "XBP1": (0.4, 1.0), "DDIT3": (2.0, 2.0)}
    for sym, (tp, tm) in fast.items():
        hit = g["symbol"] == sym
        g.loc[hit, ["t_half_prot", "t_half_mrna"]] = (tp, tm)
        g.loc[hit, ["t_half_prot_src", "t_half_mrna_src"]] = "literature (curated)"

    mu = np.log(2) / PHYS["doubling_h"]
    g["d_m"] = np.log(2) / g["t_half_mrna"]
    g["d_p"] = np.log(2) / g["t_half_prot"]
    # Reconcile the three measurements with a physical bound: one mRNA cannot make more than ~1000 proteins/h
    # (ribosome spacing). Where abundance x turnover implies more, RNA-seq under-counted a stable protein's
    # transcript (raise mRNA), or mass spectrometry over-counted a short-lived regulator (lower protein).
    need = g["protein"] * (g["d_p"] + mu) / g["mrna"].clip(lower=1e-12)
    over = (g["mrna"] > 0) & (need > K_TL_MAX)
    short = over & (g["t_half_prot"] < 3.0)
    g.loc[short, "protein"] = g.loc[short, "mrna"] * K_TL_MAX / (g.loc[short, "d_p"] + mu)
    g.loc[short, "protein_src"] = g.loc[short, "protein_src"] + " (capped by translation bound)"
    lift = over & ~short
    g.loc[lift, "mrna"] = g.loc[lift, "protein"] * (g.loc[lift, "d_p"] + mu) / K_TL_MAX
    g.loc[lift, "mrna_src"] = g.loc[lift, "mrna_src"] + " (raised to translation bound)"
    g["k_tx"] = g["mrna"] * (g["d_m"] + mu)
    g["k_tl"] = np.where(g["mrna"] > 0, g["protein"] * (g["d_p"] + mu) / g["mrna"].clip(lower=1e-12), 0.0)
    g["expressed"] = g["mrna"] > 0
    g.to_csv(cache)
    return g


# ----------------------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------------------
def signaling_edges(min_curation: int = 2) -> pd.DataFrame:
    """Signed, directed protein-protein causal interactions from OmniPath (literature curated)."""
    df = pd.read_csv(fetch("omnipath_sig.tsv"), sep="\t")
    df = df[(df["consensus_direction"]) & (df["curation_effort"] >= min_curation)]
    df = df[df["consensus_stimulation"] ^ df["consensus_inhibition"]]
    out = pd.DataFrame({"source": df["source_genesymbol"], "target": df["target_genesymbol"],
                        "sign": np.where(df["consensus_stimulation"], 1, -1), "effort": df["curation_effort"]})
    out = out[~out["source"].str.contains("_|COMPLEX", regex=True) & ~out["target"].str.contains("_|COMPLEX", regex=True)]
    return out[out["source"] != out["target"]].drop_duplicates(["source", "target"]).reset_index(drop=True)


def regulon_edges() -> pd.DataFrame:
    """Signed TF -> target-gene interactions from CollecTRI."""
    df = pd.read_csv(fetch("collectri.tsv"), sep="\t")
    df = df[df["consensus_stimulation"] ^ df["consensus_inhibition"]]
    out = pd.DataFrame({"tf": df["source_genesymbol"], "target": df["target_genesymbol"],
                        "sign": np.where(df["consensus_stimulation"], 1, -1)})
    out = out[~out["tf"].str.contains("_|COMPLEX", regex=True)]
    return out.drop_duplicates(["tf", "target"]).reset_index(drop=True)


def complexes(min_sources: int = 2) -> pd.DataFrame:
    df = pd.read_csv(fetch("complexes.tsv"), sep="\t")
    df["n_sources"] = df["sources"].astype(str).str.count(";") + 1
    df["members"] = df["components_genesymbols"].astype(str).str.split("_")
    return df[df["n_sources"] >= min_sources][["name", "members", "stoichiometry", "sources", "n_sources"]]


def essential_sets() -> tuple[set, set]:
    """Hart et al. core-essential (CEGv2) and non-essential (NEGv1) reference gene sets."""
    ceg = set(pd.read_csv(fetch("CEGv2.txt"), sep="\t").iloc[:, 0].astype(str))
    neg = set(pd.read_csv(fetch("NEGv1.txt"), sep="\t").iloc[:, 0].astype(str))
    return ceg, neg
