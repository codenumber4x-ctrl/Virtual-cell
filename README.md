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
`python run_screen.py` runs the genome-wide knock-out screen (about 1.5 h on 4 cores).

## Results in brief
RESULTS_PLACEHOLDER

## Known limitations (short list; full ledger in notebook 7)

* About half of the expressed genes are simulated as molecules only; knocking them out changes nothing.
* Signalling and regulatory strengths are set by rule, not fitted; transcriptional fold changes are under-estimated.
* Metabolism is steady-state FBA: no metabolite concentrations or allostery, and it over-predicts metabolic
  flexibility (growth without glucose, no response to methotrexate).
* Two thirds of half-lives are median defaults; module thresholds are textbook time scales, not HeLa fits.
* No space, no organelle dynamics, no explicit post-translational states, no epigenome.
* Only HeLa has been run end to end; the data layer builds seven other HPA cell lines but they are untested.
* Validation uses pan-cell-line core-essential genes; HeLa-specific DepMap scores could not be downloaded here.
