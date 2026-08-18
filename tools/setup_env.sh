#!/usr/bin/env bash
# Create the Python + Scala notebook environment for this fork.
#
#   bash tools/setup_env.sh
#
# Produces:
#   .venv/                                  Python 3.13 env (uv) + editable tools/
#   .venv/share/jupyter/kernels/scala3      Almond kernel, Scala 3.8.1
#
# Safe to re-run: uv sync is incremental and the kernel install is idempotent.
#
# Requires uv, coursier (cs) and scala-cli on PATH.
#
# Two things here are load-bearing and easy to get wrong:
#   * A JDK 17 is provisioned via coursier rather than using the system JDK,
#     because Almond's default interrupt path uses APIs removed after 17.
#   * dimwit/deepwit reach JAX through ScalaPy, which embeds a CPython in the
#     JVM. That interpreter must be THIS venv (it has jax + einops), located
#     via the -D properties computed by python-native-libs below.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

ALMOND_VERSION=0.14.5
# Must be >= the Scala version dimwit/deepwit were built with (3.8.1): a Scala
# 3.3 compiler cannot read TASTy produced by 3.8.
SCALA_VERSION=3.8.1
KERNEL_DIR="$ROOT/.venv/share/jupyter/kernels"
KERNEL_ID=scala3

echo "==> Python environment"
# uv sync builds .venv from pyproject.toml + uv.lock. The root is declared
# `package = false`, so uv never builds this directory -- setup.py, which
# defines the shipped `d2l` library with its 2023 pins, is left alone. tools/
# is installed as an editable package so `%load_ext scala_magic` resolves from
# any notebook directory.
(cd "$ROOT" && unset VIRTUAL_ENV && uv sync)

# The `d2l` library itself lives in ./d2l and is deliberately not pip-installed
# (setup.py's 2023 pins would fight the versions above). A .pth file puts the
# repo root on sys.path instead, so `from d2l import torch as d2l` resolves from
# any notebook directory -- a Jupyter kernel's cwd is the notebook's own folder,
# not the repo root.
"$ROOT/.venv/bin/python" - "$ROOT" <<'PTH'
import sys, sysconfig, pathlib
pth = pathlib.Path(sysconfig.get_paths()["purelib"]) / "_d2l_repo_root.pth"
pth.write_text(sys.argv[1] + "\n")
print(f"    wrote {pth}")
PTH

echo "==> JDK 17"
eval "$(cs java --jvm temurin:17 --env)"

echo "==> ScalaPy properties for $ROOT/.venv/bin/python"
PROPS_SCRIPT=$(mktemp -t props).sc
cat > "$PROPS_SCRIPT" <<'SC'
//> using scala 3.3.7
//> using dep ai.kien::python-native-libs:0.2.5
ai.kien.python.Python(args(0)).scalapyProperties.get.foreach { case (k, v) => println(s"-D$k=$v") }
SC
SCALAPY_OPTS=$(scala-cli run --quiet "$PROPS_SCRIPT" -- "$ROOT/.venv/bin/python" | grep '^-D')
rm -f "$PROPS_SCRIPT"
echo "$SCALAPY_OPTS" | sed 's/^/    /'

echo "==> Almond kernel: Scala $SCALA_VERSION"
# jitpack is required: almond depends on com.github.jupyter:jvm-repr, which is
# not published to Maven Central.
cs launch --use-bootstrap -r jitpack "sh.almond:scala-kernel_$SCALA_VERSION:$ALMOND_VERSION" -- \
  --install --force --copy-launcher \
  --id "$KERNEL_ID" --display-name "Scala 3 (Almond)" \
  --jupyter-path "$KERNEL_DIR" \
  --extra-repository jitpack >/dev/null

# Rewrite the generated kernel.json: pin the JDK, add the ScalaPy properties,
# and keep only the flags that mean anything at kernel run time. ivy2Local lets
# `import $ivy` resolve dimwit/deepwit from `sbt publishLocal` output.
"$ROOT/.venv/bin/python" - "$JAVA_HOME" "$KERNEL_DIR/$KERNEL_ID" "$SCALAPY_OPTS" <<'PY'
import json, sys
java_home, kdir, scalapy_opts = sys.argv[1], sys.argv[2], sys.argv[3]
argv = [f"{java_home}/bin/java", "-XX:MaxRAMPercentage=75"]
argv += [o for o in scalapy_opts.splitlines() if o.strip()]
argv += ["-cp", f"{kdir}/launcher.jar", "coursier.bootstrap.launcher.Launcher",
         "--extra-repository", "jitpack",
         "--extra-repository", "ivy2Local",
         "--use-thread-interrupt",
         "--connection-file", "{connection_file}"]
json.dump({"argv": argv, "display_name": "Scala 3 (Almond)", "language": "scala"},
          open(f"{kdir}/kernel.json", "w"), indent=2)
print(f"    wrote {kdir}/kernel.json")
PY

echo "==> Notebook type printer: tools/tprint -> ~/.ivy2/local"
# pprint's TPrint prints the whole value's type into any node its walker has no
# case for, which mangles inferred dimwit types in the REPL.  This tiny library
# ships a corrected instance; scala_magic's predef imports it into every kernel.
(cd "$ROOT/tools/tprint" && scala-cli --power publish local . >/dev/null)
echo "    published ch.contrafactus::d2l-tprint:0.1.0-SNAPSHOT"

echo
echo "==> Done. Kernels:"
"$ROOT/.venv/bin/jupyter" kernelspec list
echo
echo "Launch with:  .venv/bin/jupyter lab"
