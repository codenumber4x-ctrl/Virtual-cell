"""Collect the numbers shown on the results page into artifact/data.json (from executed notebooks + fresh runs)."""
import io, json, warnings
import nbformat, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from humancell import data
from humancell.cell import VirtualCell
from humancell.experiments import run_many
from humancell.signaling import Signaling
from humancell.metabolism import Metabolism

out = {}
g = data.build_gene_table("HeLa")

def nb_tables(name):
    """HTML tables of an executed notebook; the index becomes the first plain column."""
    nb = nbformat.read(f"notebooks/{name}.ipynb", 4); tabs = []
    for c in nb.cells:
        if c.cell_type == "code":
            for o in c.outputs:
                if "text/html" in o.get("data", {}):
                    try:
                        df = pd.read_html(io.StringIO(o.data["text/html"]))[0]
                    except Exception:
                        continue
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [a if not str(a).startswith("Unnamed") else b for a, b in df.columns]
                    df = df.rename(columns={df.columns[0]: "name"})
                    tabs.append(df.where(df.notna(), None))
    return tabs

# --- data layer
e = g[g.expressed]
out["genes"] = dict(total=len(g), expressed=int(g.expressed.sum()))
frac = ((e.protein * e.mass).groupby(e.compartment).sum() / (e.protein * e.mass).sum()).sort_values(ascending=False)
out["compartments"] = [[k.replace("_", " "), round(float(v) * 100, 1)] for k, v in frac.items()]
prov = {}
for col, lab in [("mrna_src", "mRNA abundance"), ("protein_src", "protein abundance"), ("t_half_prot_src", "protein half-life"), ("t_half_mrna_src", "mRNA half-life"), ("compartment_src", "compartment")]:
    s = e[col]; meas = s.str.contains("HPA|PaxDB HeLa|Schwan|literature") & ~s.str.contains("imputed")
    prov[lab] = round(float(meas.mean()) * 100)
out["provenance"] = prov
out["checks"] = nb_tables("01_data_and_parameters")[2].values.tolist()

# --- metabolism
met = Metabolism(g)
ref = pd.Series({k: met.v_ref[i] for k, i in met.exchange.items()}); ref = ref[ref.abs() > 0.03].sort_values()
out["exchange"] = [[k, round(float(v), 3)] for k, v in ref.items()]
t = nb_tables("02_metabolism")
out["medium"] = t[0].values.tolist(); out["medium_cols"] = ["condition"] + list(t[0].columns[1:])
out["met_stats"] = dict(reactions=met.n, enzymatic=len(met.gpr), genes=len(met.gene_ids), closed=len(met.closed))

# --- signalling heatmap
sig = Signaling(g); ones = np.ones(sig.n)
readouts = ["EGFR", "MAPK1", "FOS", "MYC", "AKT1", "FOXO3", "MTOR", "RELA", "JUN", "CASP8", "SMAD3", "STAT1", "STAT3", "TP53", "CDKN1A", "CTNNB1"]
conds = {"EGF": dict(clamp={"EGF": 1.0}), "serum-free": dict(serum=0.0), "MEK inhibitor": dict(off=["MAP2K1", "MAP2K2"]),
         "PI3K inhibitor": dict(off=["PIK3CA"]), "PTEN loss": dict(off=["PTEN"]), "TNF": dict(clamp={"TNF": 1.0}),
         "TRAIL": dict(clamp={"TNFSF10": 1.0}), "TGF-β": dict(clamp={"TGFB1": 1.0}), "IFN-γ": dict(clamp={"IFNG": 1.0}),
         "IL-6": dict(clamp={"IL6": 1.0}), "WNT3A": dict(clamp={"WNT3A": 1.0}), "DNA damage": dict(clamp={"ATM": 1.0})}
H = {}
for name, kw in conds.items():
    sig.a, sig.clamp = sig.a_ref.copy(), {}
    if "serum" in kw: sig.serum(kw["serum"])
    if "clamp" in kw: sig.set(**kw["clamp"])
    av = ones.copy()
    for s in kw.get("off", []): av[sig.pos[s]] = 0.03
    for _ in range(80): sig.step(av)
    H[name] = [round(float(np.log2(max(sig.activity(r), 1e-3))), 2) for r in readouts]
sig.a, sig.clamp = sig.a_ref.copy(), {}
out["heat"] = dict(rows=readouts, cols=list(conds), values=H)
out["sig_stats"] = dict(nodes=sig.n, edges=len(sig.edges), tfs=len(sig.tfs), reg=len(sig.reg))

# --- whole cell, deterministic
c = VirtualCell(stochastic=False); c.run(96); h = c.frame().iloc[::3]
conc = lambda k: (h[f"p:{k}"] / h["size"]) / (h[f"p:{k}"] / h["size"]).mean()
out["wt"] = dict(t=h.t.round(2).tolist(), size=h["size"].round(3).tolist(), phase=h.phase.tolist(),
                 cycE=conc("CCNE1").round(3).tolist(), cycB=conc("CCNB1").round(3).tolist(), divisions=[round(x, 1) for x in c.divisions])
out["bench"] = nb_tables("04_whole_cell_simulation")[0].values.tolist()

# --- stochastic lineages
pop = run_many({f"cell {i}": [("init", (), {"seed": i, "stochastic": True}), ("run", (110,), {})] for i in range(8)})
out["lineages"] = [dict(t=r["frame"].t.iloc[::6].round(1).tolist(), size=r["frame"]["size"].iloc[::6].round(3).tolist()) for r in pop.values()]
cyc = np.concatenate([np.diff([0] + r["divisions"]) for r in pop.values()])
out["cycles"] = [round(float(x), 1) for x in cyc]
out["fates_stoch"] = pd.Series([r["summary"]["fate"] for r in pop.values()]).value_counts().to_dict()

# --- experiments (from executed notebook 5)
t5 = nb_tables("05_in_silico_experiments")
ft = t5[0]; out["experiments"] = ft[["name", "fate", "hours", "divisions", "phase", "growth_per_h"]].values.tolist()
out["genotype"] = t5[1].values.tolist(); out["genotype_cols"] = ["experiment"] + list(t5[1].columns[1:])
nb = nbformat.read("notebooks/05_in_silico_experiments.ipynb", 4)

# --- time courses for four experiments
rec = {"oligomycin": [("run", (6,), {}), ("drug", ("oligomycin",), {}), ("run", (42,), {})],
       "tunicamycin": [("run", (6,), {}), ("drug", ("tunicamycin",), {}), ("run", (42,), {})],
       "radiation 6 Gy": [("run", (6,), {}), ("irradiate", (6,), {}), ("run", (42,), {})],
       "paclitaxel": [("run", (6,), {}), ("drug", ("paclitaxel",), {}), ("run", (42,), {})]}
r = run_many(rec, stochastic=False, met_interval=1800.0)
tc = {}
for k, v in r.items():
    f = v["frame"].iloc[::3]
    tc[k] = dict(t=f.t.round(2).tolist(), glc=(-f.q_glc).round(3).tolist(), lac=f.q_lac.round(3).tolist(), U=f.U.round(3).tolist(),
                 isr=f.isr.round(3).tolist(), D=f.D.round(1).tolist(), C=f.C.round(3).tolist(),
                 bip=((f["p:HSPA5"] / f["size"]) / (f["p:HSPA5"] / f["size"]).iloc[5]).round(3).tolist())
out["courses"] = tc

# --- coverage tiers and knock-out screen (partial results are labelled as such)
mf = pd.read_csv("results/modelled_function.csv", index_col=0)
tier = np.select([~mf.expressed, mf.kinetic_module, mf.machine, mf.metabolism, mf.signalling | mf.transcription_factor],
                 ["not expressed in HeLa", "kinetic module", "molecular machine", "metabolism", "signalling / regulation"], default="expression dynamics only")
out["tiers"] = pd.Series(tier).value_counts().to_dict()
ko = pd.read_csv("results/ko_screen.csv").drop_duplicates("gene").set_index("gene")
ceg, neg = data.essential_sets()
T = mf.drop_duplicates("symbol").set_index("symbol"); n_fun = int((T.expressed & T.any_function).sum())
sim = T.loc[T.index.intersection(ko.index)].copy(); sim["fitness"] = ko.fitness.reindex(sim.index); sim["ess"] = sim.fitness < 0.5
layers = {}
for col, lab in [("machine", "molecular machines"), ("kinetic_module", "kinetic modules"), ("metabolism", "metabolism"), ("signalling", "signalling network"), ("transcription_factor", "transcription factors")]:
    s = sim[sim[col] & sim.index.isin(ceg)]
    layers[lab] = [int(len(s)), int(s.ess.sum())]
out["screen"] = dict(simulated=int(len(ko)), functional=n_fun, lethal=int((ko.fitness < 0.5).sum()),
                     ceg_sim=int(sim.index.isin(ceg).sum()), ceg_ess=int((sim.ess & sim.index.isin(ceg)).sum()),
                     other_sim=int((~sim.index.isin(ceg)).sum()), other_ess=int((sim.ess & ~sim.index.isin(ceg)).sum()),
                     fates=ko.fate.str.replace(r" \(.*", "", regex=True).value_counts().to_dict(), layers=layers, ceg_total=len(ceg))
json.dump(out, open("artifact/data.json", "w"))
print("ok", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in out.items()})
