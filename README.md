# Virtual-cell: a genome-wide, multi-process simulation of a human cell

An executable model of a proliferating human cell (HeLa by default) in which **every protein-coding gene is
simulated** and the major cellular processes are coupled through one shared state. It grows, replicates,
divides, responds to nutrients, oxygen, growth factors, death ligands, radiation, drugs and gene knock-outs,
and reports its own coverage and gaps.

It is a research instrument, not a finished digital twin: about half of the expressed genes have a modelled
function, most kinetic constants are inferred rather than measured, and there is no spatial resolution.
Notebook 7 lists exactly what is and is not in the model.

## What is inside

| Layer | Formalism | Scale | Data |
|---|---|---|---|
| Transcription, translation, turnover | stochastic birth–death per gene | 20,162 genes | HPA RNA-seq, PaxDB proteome, Schwanhäusser half-lives |
| Molecular machines | capacity of the scarcest subunit, with reserve | 40 machines, ~930 genes | curated composition |
| Metabolism | enzyme-constrained flux balance | 12,748 reactions, 8 compartments | Human-GEM |
| Signalling | signed network, linearised at the reference state | 5,144 proteins | OmniPath + 208 backbone edges |
| Gene regulation | TF-activity regulons | 1,163 TFs, 43k edges | CollecTRI |
| Stress and fate modules | reduced ODEs | energy/AMPK/mTORC1, ISR/UPR, hypoxia, redox, DNA damage/p53, ions/Vm, Ca²⁺, apoptosis, necrosis | textbook kinetics |
| Cell cycle | checkpoint-gated hybrid automaton | G1/S/G2/M, replication, division, lineages | — |

The design follows the Karr 2012 / Vivarium principle: each process in the mathematics its data supports,
all reading and writing one state on a short time step.

## Notebooks

| | |
|---|---|
| `00_landscape` | earlier whole-cell efforts, why they worked, where they stop, and the design drawn from them |
| `01_data_and_parameters` | the per-gene parameter table, provenance, physical consistency checks |
| `02_metabolism` | calibration, environment predictions, metabolic gene deletions |
| `03_signalling_and_regulation` | pathway responses to 12 stimuli and inhibitors |
| `04_whole_cell_simulation` | growth and division, stochastic lineages, protein noise |
| `05_in_silico_experiments` | 50 experiments: nutrients, ligands, radiation, 26 drugs, knock-outs, genotype swaps, dose–response |
| `06_validation` | genome-wide knock-out screen against CRISPR core-essential genes |
| `07_gap_ledger` | coverage by fidelity tier, missing processes, parameter provenance, roadmap |

Notebooks are generated from `notebooks/src/*.py` and executed by `python tools/nbbuild.py`.

## Quick start

```bash
pip install -r requirements.txt
python -c "from humancell import data; data.fetch_all()"      # ~0.5 GB of public data into ./data
```

```python
from humancell.cell import VirtualCell

cell = VirtualCell(stochastic=True, seed=1)     # a newborn G1 HeLa cell
cell.run(24)                                    # hours
cell.drug("trametinib")                         # or .knockout("TP53"), .set_medium(glucose=0), .irradiate(4),
cell.run(48)                                    #    .ligand(TNF=1.0), .serum(0), .overexpress("MYC") ...
print(cell.summary())
cell.frame()                                    # time course as a DataFrame; cell.m / cell.p = all mRNA / protein copies
```

A simulated day takes about 6 s deterministic, 20 s stochastic, on one core.
`python run_screen.py` runs the genome-wide knock-out screen (several hours on 4 cores; checkpointed and resumable).

## Results in brief
* **Unperturbed cell.** Deterministic HeLa cycle 23.3 h (G1 12.6, S 8.1, G2 3.3, M 1.0). Eight stochastic cells
  all survive 110 h and divide every 20-22 h (s.d. 3.5-5 h).
* **Coverage.** 12,665 genes expressed; 6,122 have a modelled function (metabolism, machine, signalling,
  transcription factor or kinetic module). The rest are present as molecules only.
* **51 perturbation experiments** (notebook 5). Reproduced: HeLa resistance to palbociclib and nutlin and their
  rescue when p53/RB are restored, TNF survival vs TNF + cycloheximide apoptosis, TRAIL apoptosis, glycolytic
  shift under oligomycin, mitotic arrest then death under paclitaxel, radiation arrest. Missed: methotrexate
  has no effect.
* **Knock-out screen** (notebook 6). Incomplete: 2,400 of 6,122 functional genes simulated before the run-time
  limit; `python run_screen.py` resumes it. All 515 modelled core-essential genes (Hart CEGv2) are included:
  344 are called essential (67%), as are 343 of the 1,885 other genes simulated (18%); none of 21 simulated
  non-essential reference genes. AUC among simulated genes 0.76. Counting all 662 expressed core-essential
  genes, recall is 0.52, because 147 have no modelled function.
* **Baseline check.** Protein abundance alone separates core-essential from non-essential reference genes
  better than the model does (AUC 0.99 vs 0.79), because the reference negatives are mostly unexpressed. The
  model is only ahead among genes with a modelled function (0.83 vs 0.80), and that figure is optimistic
  while the screen is incomplete.

## Known limitations (short list; full ledger in notebook 7)

* About half of the expressed genes are simulated as molecules only; knocking them out changes nothing.
* Signalling and regulatory strengths are set by rule, not fitted; transcriptional fold changes are under-estimated.
* Metabolism is steady-state FBA: no metabolite concentrations or allostery, and it over-predicts metabolic
  flexibility (growth without glucose, no response to methotrexate).
* Two thirds of half-lives are median defaults; module thresholds are textbook time scales, not HeLa fits.
* No space, no organelle dynamics, no explicit post-translational states, no epigenome.
* Only HeLa has been run end to end; the data layer builds seven other HPA cell lines but they are untested.
* Validation uses pan-cell-line core-essential genes; HeLa-specific DepMap scores could not be downloaded here.

## 3D viewer

`twin3d/` holds a three.js page that plays the simulation back as a 3D cell under nine conditions.
`PYTHONPATH=. python3 twin3d/simulate.py` runs the model and writes `data.json`; `python3 twin3d/build.py`
embeds it into `index.html`. Organelle geometry is schematic; the state driving it is model output.
