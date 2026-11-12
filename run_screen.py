"""Genome-wide in-silico knock-out screen (every expressed gene with a modelled function). ~1.5 h on 4 cores."""
import sys
from humancell.cell import VirtualCell
from humancell.screen import knockout_screen, modelled_function

if __name__ == "__main__":
    cell = VirtualCell(stochastic=False)
    mf = modelled_function(cell)
    mf.to_csv("results/modelled_function.csv")
    genes = mf[mf.expressed & mf.any_function]["symbol"].drop_duplicates().tolist()
    if len(sys.argv) > 1:
        genes = genes[: int(sys.argv[1])]
    df = knockout_screen(genes, hours=72.0)
    df.to_csv("results/ko_screen.csv", index=False)
    print(df.attrs["wild_type"], len(df), (df.fitness < 0.5).sum())
