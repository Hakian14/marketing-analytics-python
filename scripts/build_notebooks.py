"""
Execute the notebook sources in notebooks/src/*.py (Jupytext "percent" format)
and write fully rendered .ipynb files to notebooks/.

    python scripts/build_notebooks.py            # build all
    python scripts/build_notebooks.py 02         # build only 02_*

Why this exists: the notebooks are kept as plain .py files (diff-able, lint-able,
runnable as scripts) and rendered to .ipynb with outputs so they read well on
GitHub. The runner needs only the standard library + the notebook's own imports;
no Jupyter installation is required to rebuild them.

Cell syntax (same as VS Code / Jupytext):
    # %% [markdown]
    # Markdown text, one leading "# " per line
    # %%
    code cell
"""

from __future__ import annotations

import base64
import contextlib
import glob
import io
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "notebooks", "src")
OUT = os.path.join(ROOT, "notebooks")


def parse_percent(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def html_of(obj):
    if obj is None:
        return None
    rep = getattr(obj, "_repr_html_", None)
    if callable(rep):
        try:
            return rep()
        except Exception:  # pragma: no cover
            return None
    return None


def run_code(code: str, ns: dict):
    """Execute a cell; return (stdout_text, last_expr_value, error_text, figures_png)."""
    import ast

    stdout = io.StringIO()
    value, err = None, None
    try:
        tree = ast.parse(code)
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, "<cell>", "exec"), ns)
            if last_expr is not None:
                value = eval(compile(last_expr, "<cell>", "eval"), ns)
    except Exception:
        err = traceback.format_exc()
    figs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        figs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        plt.close(fig)
    return stdout.getvalue(), value, err, figs


def build(path: str) -> bool:
    name = os.path.splitext(os.path.basename(path))[0]
    cells = parse_percent(open(path, encoding="utf-8").read())
    ns = {"__name__": "__main__", "__file__": path}
    os.chdir(ROOT)
    nb_cells, ok = [], True
    for i, (kind, body) in enumerate(cells):
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
            continue
        out, value, err, figs = run_code(body, ns)
        outputs = []
        if out:
            outputs.append({"output_type": "stream", "name": "stdout", "text": out})
        if value is not None:
            data = {"text/plain": repr(value)}
            h = html_of(value)
            if h:
                data["text/html"] = h
            outputs.append({"output_type": "execute_result", "metadata": {}, "execution_count": i + 1, "data": data})
        for png in figs:
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": png}})
        if err:
            ok = False
            print(f"  ERROR in {name} cell {i + 1}:\n{err}", file=sys.stderr)
            outputs.append({"output_type": "stream", "name": "stderr", "text": err})
        nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": i + 1, "source": body, "outputs": outputs})
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = os.path.join(OUT, name + ".ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  {'ok ' if ok else 'ERR'} {os.path.relpath(out_path, ROOT)}  ({len(nb_cells)} cells)")
    return ok


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    sources = sorted(glob.glob(os.path.join(SRC, f"{prefix}*.py")))
    if not sources:
        print("no notebook sources found", file=sys.stderr)
        return 1
    sys.path.insert(0, ROOT)
    results = [build(p) for p in sources]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
