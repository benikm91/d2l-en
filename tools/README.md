# Python + Scala notebooks for this d2l fork

Each d2l example runs next to a [dimwit](https://github.com/dimwit-dev/dimwit)
equivalent in **one** notebook, with the eventual aim of feeding the Scala
variants back into the book build as an extra tab on <https://d2l.ai>.

## Setup

```bash
bash tools/setup_env.sh      # needs uv, cs (coursier) and scala-cli on PATH
.venv/bin/jupyter lab
```

Python dependencies live in [`../pyproject.toml`](../pyproject.toml), pinned by
`uv.lock`; `uv sync` alone rebuilds just the Python half. The root is declared
`package = false` so uv never builds this directory as a package -- `setup.py`
still defines the shipped `d2l` library, and its 2023 pins would otherwise
fight with the versions the notebooks need.

The `d2l` library therefore stays un-installed, but chapters still say
`from d2l import torch as d2l`, and a Jupyter kernel's working directory is the
notebook's own folder rather than the repo root. `setup_env.sh` bridges that by
dropping a `_d2l_repo_root.pth` into `site-packages` pointing at the repo root
-- the import resolves from any chapter directory, without setup.py ever being
built. `torchvision` is a dependency for the same reason: `d2l/torch.py` imports
it at module scope, so even a chapter that never touches an image needs it.

## How one notebook runs two languages

Jupyter binds a notebook to exactly one kernel, so a genuinely bi-lingual
notebook is not possible directly. Instead the notebook runs on the **Python**
kernel, and [`scala_magic.py`](scala_magic.py) registers a `%%scala` cell magic
that forwards the cell to an **Almond** (Scala) kernel started as a child
process and kept alive for the session. Outputs -- stdout, values, errors, rich
HTML/image output -- are relayed back into the Python notebook's output area.
Because the Almond kernel is persistent, Scala `val`s defined in one cell are
visible in the next, exactly like Python state.

Line magics: `%scala_info` (versions), `%scala_reset` (drop all Scala state),
`%scala_kernel <id>` (switch kernelspec).

## The stack

| piece | version | why |
|---|---|---|
| Almond kernel | Scala **3.8.1** | dimwit/deepwit are built with 3.8.1, and a 3.3 compiler cannot read 3.8 TASTy |
| JDK | **17** (via coursier) | Almond's default interrupt path uses APIs removed after 17 |
| CPython | **3.13**, this venv | dimwit reaches JAX through ScalaPy, which embeds a CPython in the JVM |

`setup_env.sh` computes the ScalaPy `-D` properties with `python-native-libs`
(the same mechanism dimwit's own `build.sbt` uses) and bakes them into
`kernel.json`, so the embedded interpreter is `.venv/bin/python` -- which is why
`jax` and `einops` are dependencies of this project and not just of dimwit.

## Using dimwit / deepwit / plotwit from a notebook

They are resolved from `~/.ivy2/local` (`sbt publishLocal`); the kernel is
configured with the `ivy2Local` repository for exactly this:

```scala
%%scala
import $ivy.`ch.contrafactus::dimwit-core:0.1.0-SNAPSHOT`
dimwit.initialize()
```

### Plots

Figures come from [plotwit](https://github.com/dimwit-dev/plotwit), which builds
a Vega-Lite spec from dimwit tensors -- the Scala counterpart to `d2l.plot`.
JupyterLab draws the spec with its own bundled vega-embed, so the figure is
interactive and nothing is pre-rendered into the notebook:

```scala
%%scala
import $ivy.`ch.contrafactus::plotwit-core:0.1.0-SNAPSHOT`
import plotwit.{Grid as _, *}
import viz.PlotTargets.almond

plotwit.display(plots.linePlot(xs, ys, names, _.encoding.x.title := "x"))
```

Three papercuts, two of which that cell has to dodge by hand:

* plotwit exports a type named `Grid`, which shadows a `Grid` axis defined in an
  earlier cell -- hence `{Grid as _, *}`.
* `display` is ambiguous: Almond's predef already imports its own
  `publish.display`, so plotwit's has to be qualified.
* `viz.PlotTargets.almond` publishes every chart under the **Vega** mime with
  plotwit's Vega-Lite **v6** schema, and Almond's `DisplayData` can only carry
  strings, whereas a `+json` payload travels as an object. All three mismatches
  are repaired in `scala_magic._adapt_vega`, which parses the payload, re-keys
  it to `application/vnd.vegalite.v5+json` and pins the schema to v5 -- the
  newest JupyterLab 4.6 registers a renderer for. Without that the cell runs
  clean and the output area stays silently empty.

`viz.PlotTargets.almond_js` cannot work here whatever the bridge does: it emits
`application/javascript`, which JupyterLab 4 no longer executes. plotwit's
`displayAsImage` is the fallback for a static PNG, but it shells out to `vl2png`
(`npm i -g vega-cli vega-lite`), which the MIME route does not need.

### Why every dimwit cell is wrapped in an `object`

dimwit's documented entry point is `import dimwit.*`. **That does not work at
cell top level in Almond.** Ammonite (Almond's engine) replays each cell's
imports into the next one by enumerating the names in scope, and it does not
backtick operator names. dimwit's package exports members literally named `*`,
`+`, `<=` (via `export TensorOps.*`), so the replay emits a bare `*`, Scala 3
reads it as a wildcard, and every subsequent cell fails with *"named imports
cannot follow wildcard imports"*.

Confirmed still present in Ammonite 3.0.9 (newer than the 3.0.8 Almond pins).
Two non-fixes, for the record:

* Splitting into `import dimwit.tensor.TensorOps.*` plus
  `import dimwit.tensor.ValueOps.*` makes the replay parse, but then `ValueOps`'
  scalar `+` shadows the general tensor `+` and rank>0 arithmetic stops
  compiling. Import order does not matter.
* An aggregator `object DW { export ... }` re-exports the same operator names
  and hits the identical replay bug.

What does work is keeping `import dimwit.*` inside a per-cell object, so those
names never enter cell scope:

```scala
%%scala
object Scalars:
  import dimwit.*
  val x = Tensor0(3.0f)
  val y = Tensor0(2.0f)
  val out = (x + y, x * y, x / y, x.pow(y))
Scalars.out
```

Later cells reach earlier values by qualifying (`Scalars.x`) and re-import
`dimwit.*` inside their own object. Verbose for a textbook -- see the open
question at the end of this file.

## Generating a notebook from a chapter

```bash
.venv/bin/python tools/md2ipynb.py chapter_preliminaries/linear-algebra.md
```

[`md2ipynb.py`](md2ipynb.py) picks one framework tab out of the d2lbook
markdown and emits a plain notebook with an empty `%%scala` companion cell
after every code cell. It refuses to overwrite an existing notebook
(hand-written Scala cells would be lost); `--force` regenerates anyway,
`--tab jax` picks another framework, `--no-scala-stubs` skips the companions.

## Open question

The per-cell `object` wrapper is a workaround, not a design. Better options, in
rough order of preference:

1. Have `scala_magic` wrap cells automatically and lift the object's members
   back into scope (`import cellN.*` only replays the user's own names, which
   are harmless). Invisible to the book author; needs care around Scala's lazy
   object initialisation and around picking out a cell's final expression.
2. Avoid package-level operator-named exports in dimwit itself.
3. Fix the import replay upstream in Ammonite.
