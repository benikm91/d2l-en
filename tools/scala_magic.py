"""Run Scala cells inside a Python Jupyter notebook.

Registers a ``%%scala`` cell magic that forwards the cell to a persistent
Almond (Scala) kernel started as a child process.  Because the kernel is
persistent, Scala state carries across cells exactly like Python state does,
so a notebook can interleave ``torch`` cells and Scala cells and compare them
side by side.

Usage in a notebook::

    %load_ext scala_magic

    %%scala
    val x = 3.0
    x + 1

Extra line magics: ``%scala_reset`` (restart the Scala kernel),
``%scala_info`` (versions / status), ``%scala <code>`` (one-liner).
"""

from __future__ import annotations

import json
import os
import queue
import sys

from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.display import publish_display_data

#: kernelspec name to launch; override with the D2L_SCALA_KERNEL env var.
#: "scala3" is Scala 3.3 LTS, "scala213" is Scala 2.13 (for 2.13-only libraries
#: such as Breeze).  Both are installed under .venv/share/jupyter/kernels.
KERNEL_NAME = os.environ.get("D2L_SCALA_KERNEL", "scala3")

#: seconds to wait for the kernel to come up, and for any single cell to finish
STARTUP_TIMEOUT = float(os.environ.get("D2L_SCALA_STARTUP_TIMEOUT", 300))
EXEC_TIMEOUT = float(os.environ.get("D2L_SCALA_EXEC_TIMEOUT", 600))

#: mime types we hand to the Python frontend as-is, most specific first.  The
#: Vega ones let a Scala cell publish a spec that JupyterLab draws with its own
#: bundled vega-embed -- an interactive figure rather than a static image.
_RICH_MIMES = ("application/vnd.vegalite.v5+json", "application/vnd.vega.v5+json",
               "text/html", "image/svg+xml", "image/png", "image/jpeg",
               "application/javascript", "text/latex", "text/markdown")

#: JupyterLab 4.6 registers renderers for Vega-Lite v3/v4/v5 and Vega v5, and
#: nothing newer.  Two mismatches have to be bridged for a plotwit figure to
#: come out of `display(...)`, and both are fixed up in _publish below:
#:   * dedav4s' almond PlotTarget publishes every chart under the *Vega* mime,
#:     so a Vega-Lite spec reaches the frontend's Vega renderer and fails;
#:   * plotwit's spec templates declare the Vega-Lite v6 schema.
_VEGALITE_MIME = "application/vnd.vegalite.v5+json"
_VEGALITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"

#: run in the Almond kernel right after startup, before any user cell.  pprint's
#: type printer mis-renders inferred types -- its structural walker prints the
#: whole value's type into any node it has no case for, so `t0.pow(t0)` comes out
#: as `Tensor[Tensor[EmptyTuple, Float32], Float32]`.  tools/tprint ships a
#: corrected `TPrint`, and importing it into cell scope wins over pprint's, which
#: is deliberately low priority.  Ammonite replays the import into every later
#: cell.  Set D2L_SCALA_NO_PREDEF=1 to skip (e.g. if the jar is not published).
PREDEF = """import $ivy.`ch.contrafactus::d2l-tprint:0.1.0-SNAPSHOT`
import d2l.TPrintNice.given"""


class ScalaKernel:
    """A lazily started Almond kernel plus a blocking client."""

    def __init__(self, kernel_name: str = KERNEL_NAME):
        self.kernel_name = kernel_name
        self._km = None
        self._kc = None

    @property
    def started(self) -> bool:
        return self._km is not None

    def start(self):
        if self._km is not None:
            return
        from jupyter_client.manager import KernelManager

        sys.stderr.write(f"[scala] starting {self.kernel_name} kernel ...\n")
        sys.stderr.flush()
        km = KernelManager(kernel_name=self.kernel_name)
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=STARTUP_TIMEOUT)
        except Exception:
            kc.stop_channels()
            km.shutdown_kernel(now=True)
            raise
        self._km, self._kc = km, kc
        if not os.environ.get("D2L_SCALA_NO_PREDEF"):
            self._run_predef()
        sys.stderr.write("[scala] ready\n")
        sys.stderr.flush()

    def _run_predef(self):
        """Run PREDEF, surfacing only failures -- its own output is noise."""
        msg_id = self._kc.execute(PREDEF)
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=EXEC_TIMEOUT)
            except queue.Empty:
                sys.stderr.write("[scala] predef timed out\n")
                return
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            # Almond reports compile failures as stderr text, not as an error msg.
            if kind == "error":
                sys.stderr.write("[scala] predef failed: {}: {}\n".format(
                    content.get("ename", "error"), content.get("evalue", "")))
            elif kind == "stream" and content["name"] == "stderr":
                sys.stderr.write("[scala] predef: " + content["text"])
            elif kind == "status" and content["execution_state"] == "idle":
                return

    def shutdown(self):
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None:
            self._km.shutdown_kernel(now=True)
        self._km = self._kc = None

    def restart(self):
        self.shutdown()
        self.start()

    def run(self, code: str):
        """Execute ``code``, relaying every output to the Python frontend."""
        self.start()
        msg_id = self._kc.execute(code)
        errored = False
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=EXEC_TIMEOUT)
            except queue.Empty:
                sys.stderr.write("[scala] timed out waiting for the kernel\n")
                return
            except KeyboardInterrupt:
                self._km.interrupt_kernel()
                sys.stderr.write("[scala] interrupt sent\n")
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]

            if kind == "stream":
                stream = sys.stderr if content["name"] == "stderr" else sys.stdout
                stream.write(content["text"])
                stream.flush()
            elif kind in ("execute_result", "display_data"):
                self._publish(content["data"], content.get("metadata", {}))
            elif kind == "error":
                errored = True
                # Almond omits "traceback" for compilation (as opposed to
                # runtime) failures, so fall back to ename/evalue.
                tb = content.get("traceback")
                text = "\n".join(tb) if tb else "{}: {}".format(
                    content.get("ename", "error"), content.get("evalue", ""))
                sys.stderr.write(text + "\n")
                sys.stderr.flush()
            elif kind == "status" and content["execution_state"] == "idle":
                break
        return errored

    @staticmethod
    def _publish(data: dict, metadata: dict):
        rich = {m: data[m] for m in _RICH_MIMES if m in data}
        _adapt_vega(rich)
        if rich:
            if "text/plain" in data:
                rich["text/plain"] = data["text/plain"]
            publish_display_data(rich, metadata)
        elif "text/plain" in data:
            # Almond colours text/plain with ANSI codes; Jupyter renders those
            # in stream output but not in a display_data text/plain payload.
            sys.stdout.write(data["text/plain"] + "\n")
            sys.stdout.flush()


def _adapt_vega(rich: dict):
    """Make an Almond-published Vega bundle renderable by JupyterLab, in place.

    A `+json` payload travels as a JSON *object* in the Jupyter protocol, but
    Almond's ``DisplayData`` carries ``Map[String, String]``, so it arrives here
    as a string; passed on unparsed, vega-embed reads it as a URL and the output
    stays blank.  Vega-Lite specs are then re-keyed onto the Vega-Lite mime and
    pinned to the newest schema the frontend knows (see the constants above).
    """
    for mime in ("application/vnd.vega.v5+json", _VEGALITE_MIME):
        payload = rich.get(mime)
        if payload is None:
            continue
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                continue
        if not isinstance(payload, dict):
            continue
        if "vega-lite" in payload.get("$schema", ""):
            del rich[mime]
            rich[_VEGALITE_MIME] = {**payload, "$schema": _VEGALITE_SCHEMA}
        else:
            rich[mime] = payload


_KERNEL = ScalaKernel()


@magics_class
class ScalaMagics(Magics):

    @cell_magic
    def scala(self, line, cell):
        """Run the cell body in the persistent Scala kernel."""
        code = cell if not line.strip() else line + "\n" + cell
        _KERNEL.run(code)

    @line_magic("scala")
    def scala_line(self, line):
        """Run a single line of Scala."""
        if line.strip():
            _KERNEL.run(line)

    @line_magic
    def scala_reset(self, line):
        """Restart the Scala kernel, discarding all Scala state."""
        _KERNEL.restart()

    @line_magic
    def scala_info(self, line):
        """Print Scala / JVM / kernel information."""
        _KERNEL.run(
            'println("kernel : ' + _KERNEL.kernel_name + '")\n'
            'println("scala  : " + scala.util.Properties.versionNumberString)\n'
            'println("java   : " + System.getProperty("java.version"))'
        )

    @line_magic
    def scala_kernel(self, line):
        """Switch to another Scala kernelspec, e.g. `%scala_kernel scala213`."""
        name = line.strip()
        if not name:
            print(_KERNEL.kernel_name)
            return
        _KERNEL.shutdown()
        _KERNEL.kernel_name = name
        _KERNEL.start()


def load_ipython_extension(ipython):
    ipython.register_magics(ScalaMagics)


def unload_ipython_extension(ipython):
    _KERNEL.shutdown()
