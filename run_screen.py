"""Genome-wide in-silico knock-out screen: every expressed gene with a modelled function.

Results are checkpointed to results/ko_screen.csv every 200 genes, so the run can be interrupted and resumed.
Benchmark genes (core-essential, non-essential, machines, modules) are simulated first.
"""
from humancell import data
from humancell.cell import VirtualCell
from humancell.screen import knockout_screen, modelled_function

if __name__ == "__main__":
    cell = VirtualCell(stochastic=False)
    mf = modelled_function(cell)
    mf.to_csv("results/modelled_function.csv")
    ceg, neg = data.essential_sets()
    f = mf[mf.expressed & mf.any_function].drop_duplicates("symbol")
    rank = (~f.symbol.isin(ceg | neg)).astype(int) * 4 + (~(f.machine | f.kinetic_module)).astype(int) * 2 + (~f.metabolism).astype(int)
    genes = f.symbol.iloc[rank.values.argsort(kind="stable")].tolist()
    df = knockout_screen(genes, checkpoint="results/ko_screen.csv")
    print(df.attrs["wild_type"], len(df), int((df.fitness < 0.5).sum()))
