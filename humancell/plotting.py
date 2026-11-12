"""One plotting style for every figure (validated categorical palette, thin marks, recessive axes)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
PHASE = {"G1": SERIES[0], "S": SERIES[1], "G2": SERIES[2], "M": SERIES[3]}


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 2.0, "lines.markersize": 6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c8c3",
        "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "font.size": 10,
    })


def phase_bands(ax, frame):
    """Shade cell-cycle phases behind a time course."""
    t, ph = frame["t"].values, frame["phase"].values
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or ph[i] != ph[start]:
            ax.axvspan(t[start], t[min(i, len(t) - 1)], color=PHASE[ph[start]], alpha=0.10, lw=0)
            start = i
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.25) for c in PHASE.values()]
    return handles, list(PHASE)
