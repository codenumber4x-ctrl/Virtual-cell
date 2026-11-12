"""Build .ipynb files from percent-format sources in notebooks/src/*.py and execute them.

    python tools/nbbuild.py            # build + execute all
    python tools/nbbuild.py 02 05      # only notebooks whose name starts with these prefixes
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks" / "src"


def build(src: pathlib.Path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.split("\n"))
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif text:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in src.read_text().split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    out = ROOT / "notebooks" / (src.stem + ".ipynb")
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
    nbformat.write(nb, out)
    print("built", out.name)


if __name__ == "__main__":
    for src in sorted(SRC.glob("*.py")):
        if len(sys.argv) == 1 or any(src.name.startswith(p) for p in sys.argv[1:]):
            build(src)
