#!/usr/bin/env python
"""Convert a d2lbook markdown chapter into a runnable Python+Scala notebook.

The d2l sources keep one markdown file per section, with framework-specific
code marked by ``%%tab pytorch`` / ``%%tab mxnet`` / ... cell magics.  This
script picks a single tab, drops the others, and emits a plain ``.ipynb`` that
runs on an ordinary IPython kernel.

It additionally scaffolds the Scala side of the book: a bootstrap cell that
loads the ``%%scala`` magic (see ``tools/scala_magic.py``), and -- unless
``--no-scala-stubs`` is given -- one empty ``%%scala`` companion cell after
every Python code cell, ready to be filled with the Scala equivalent.

    python tools/md2ipynb.py chapter_preliminaries/linear-algebra.md
"""

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

from d2lbook import notebook as d2lnb

BOOTSTRAP = """\
# Bootstrap: Python and Scala in one notebook.
# `%%scala` cells are forwarded to a persistent Almond kernel (Scala 3.8.1),
# so Scala state carries across cells just like Python state does. That kernel
# embeds this venv's CPython via ScalaPy, which is how dimwit reaches JAX.
%load_ext scala_magic
%scala_info"""

STUB_TEMPLATE = "%%scala\n// TODO: Scala equivalent of the cell above"


def convert(md_path: str, out_path: str, tab: str, scala_stubs: bool) -> str:
    with open(md_path) as f:
        nb = d2lnb.read_markdown(f.read())
    nb = d2lnb.get_tab_notebook(nb, tab=tab, default_tab=tab)

    cells = [new_code_cell(BOOTSTRAP)]
    for cell in nb.cells:
        source = cell.source.strip()
        if not source:
            continue
        if cell.cell_type == "markdown":
            cells.append(new_markdown_cell(source))
            continue
        cells.append(new_code_cell(source))
        if scala_stubs:
            cells.append(new_code_cell(STUB_TEMPLATE))

    out = nbformat.v4.new_notebook(cells=cells)
    out.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "d2l": {"source": md_path, "tab": tab},
    })
    with open(out_path, "w") as f:
        nbformat.write(out, f)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("markdown", help="path to a d2l chapter .md file")
    p.add_argument("-o", "--output", help="output .ipynb (default: alongside the .md)")
    p.add_argument("-t", "--tab", default="pytorch",
                   help="which framework tab to keep (default: pytorch)")
    p.add_argument("--no-scala-stubs", dest="scala_stubs", action="store_false",
                   help="do not insert empty %%%%scala companion cells")
    p.add_argument("-f", "--force", action="store_true",
                   help="overwrite the output notebook if it already exists")
    args = p.parse_args()

    out = args.output or os.path.splitext(args.markdown)[0] + ".ipynb"
    if os.path.exists(out) and not args.force:
        raise SystemExit(
            f"{out} already exists; hand-written Scala cells would be lost. "
            f"Pass --force to overwrite.")
    convert(args.markdown, out, args.tab, args.scala_stubs)
    nb = nbformat.read(out, as_version=4)
    code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"{args.markdown} -> {out}  ({len(nb.cells)} cells, {code} code, tab={args.tab})")


if __name__ == "__main__":
    main()
