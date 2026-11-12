"""Run the whole-cell model under several conditions and export what the 3D viewer plays back."""
import json, numpy as np
from humancell import data
from humancell.experiments import run_many, treat

SC = {
    "Unperturbed": [("run", (72.0,), {})],
    "Paclitaxel": treat("drug", "paclitaxel", 1.0),
    "TRAIL death ligand": treat("ligand", TNFSF10=1.0),
    "Radiation, 6 Gy": treat("irradiate", 6.0),
    "Oligomycin": treat("drug", "oligomycin", 1.0),
    "Glucose withdrawal": treat("set_medium", glucose=0.0),
    "Ouabain": treat("drug", "ouabain", 1.0),
    "Thapsigargin": treat("drug", "thapsigargin", 1.0),
    "Hydroxyurea": treat("drug", "hydroxyurea", 1.0),
}
NOTE = {
    "Unperturbed": "HeLa cell in full medium with serum.",
    "Paclitaxel": "Microtubule stabiliser added at 6 h. Blocks the spindle.",
    "TRAIL death ligand": "TRAIL added at 6 h. Engages death receptors and caspase-8.",
    "Radiation, 6 Gy": "6 Gy at 6 h. About 200 DNA double-strand breaks.",
    "Oligomycin": "ATP synthase inhibitor added at 6 h.",
    "Glucose withdrawal": "Glucose removed from the medium at 6 h.",
    "Ouabain": "Na/K pump inhibitor added at 6 h.",
    "Thapsigargin": "SERCA inhibitor added at 6 h. Empties the ER calcium store.",
    "Hydroxyurea": "Ribonucleotide reductase inhibitor added at 6 h. Starves DNA synthesis.",
}
COLS = ["size", "mu", "EC", "P53", "D", "C", "Vm", "Ca", "swell", "ros", "r", "q_glc", "q_lac", "q_o2", "isr", "mtor"]

if __name__ == "__main__":
    res = run_many(SC, processes=4, stochastic=False)
    out = {"scenarios": [], "cols": COLS}
    for name in SC:
        r = res[name]; f = r["frame"].iloc[::2].reset_index(drop=True)    # every 10 min
        p53_ref = f.P53.iloc[0]
        d = {"name": name, "note": NOTE[name], "perturbed_at": None if name == "Unperturbed" else 6.0,
             "t": [round(x, 3) for x in f.t], "phase": "".join({"G1": "1", "S": "S", "G2": "2", "M": "M"}.get(p, "1") for p in f.phase),
             "gen": [int(x) for x in f.generation], "fate": r["summary"]["fate"], "end": round(float(f.t.iloc[-1]), 2)}
        for c in COLS:
            v = f[c].astype(float).values
            if c == "P53": v = v / p53_ref
            d[c] = [float(f"{x:.4g}") for x in v]
        out["scenarios"].append(d)
        print(name, r["summary"]["fate"], r["summary"]["divisions"], d["end"])
    g = data.build_gene_table("HeLa"); g = g[g.expressed]
    comp = []
    for c, s in g.groupby("compartment"):
        top = s.nlargest(8, "protein")
        comp.append({"id": c, "genes": int(len(s)), "copies": float(s.protein.sum()), "mass_fg": float((s.protein * s.mass).sum() / 6.022e23 * 1e15),
                     "top": [[a, float(b)] for a, b in zip(top.symbol, top.protein)]})
    out["compartments"] = comp
    out["total_proteins"] = float(g.protein.sum()); out["genes"] = int(len(g))
    json.dump(out, open("twin3d/data.json", "w"), separators=(",", ":"))
