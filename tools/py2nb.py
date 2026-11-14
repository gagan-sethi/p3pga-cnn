"""Rebuild the .ipynb from the percent-format .py, keeping the old notebook's metadata.

Usage: python3 tools/py2nb.py p3pga_mo_nas_maize.py p3pga_mo_nas_maize.ipynb
"""
import json
import sys
from pathlib import Path

py, nb = Path(sys.argv[1]), Path(sys.argv[2])
old_meta = json.loads(nb.read_text()).get("metadata", {}) if nb.exists() else {}
cells, cur, kind = [], [], None


def flush():
    if kind is None:
        return
    lines = cur[:]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if kind == "markdown":
        lines = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in lines]
    if not lines:
        return
    src = [l + "\n" for l in lines]
    src[-1] = src[-1].rstrip("\n")
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    cells.append(c)


for line in py.read_text().splitlines():
    if line.startswith("# %%"):
        flush()
        kind, cur = ("markdown" if "[markdown]" in line else "code"), []
    else:
        if kind is None:
            kind = "code"
        cur.append(line)
flush()
nb.write_text(json.dumps({"cells": cells, "metadata": old_meta, "nbformat": 4,
                          "nbformat_minor": 5}, indent=1))
print(f"{len(cells)} cells written to {nb}")
