"""Execute the notebook, insert actual interpretations, and export HTML/Python."""
from pathlib import Path
import argparse
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
for name, directory in {
    "IPYTHONDIR": ROOT / ".cache/ipython",
    "JUPYTER_RUNTIME_DIR": ROOT / ".cache/jupyter/runtime",
    "MPLCONFIGDIR": ROOT / ".cache/matplotlib",
    "XDG_CACHE_HOME": ROOT / ".cache",
}.items():
    os.environ.setdefault(name, str(directory))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

parser = argparse.ArgumentParser()
parser.add_argument("--export-only", action="store_true", help="Export existing executed outputs without recomputing")
args = parser.parse_args()
path = ROOT / "lab1.ipynb"
notebook = nbformat.read(path, as_version=4)
started = time.monotonic()

if not args.export_only:
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"[{time.monotonic() - started:.1f}s] executing cell {cell_index + 1}", flush=True)

    client = NotebookClient(
        notebook, timeout=1800, kernel_name="aiml", allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress,
    )
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, path)
        raise

summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
full_code = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
for cell in notebook.cells:
    key = cell.metadata.get("result_key")
    if key:
        cell.source = cell.metadata["base_source"] + "\n\n" + summary["notes"][key]
    if cell.metadata.get("appendix_code"):
        cell.source = "<details>\n<summary>Развернуть полный листинг Python</summary>\n\n```python\n" + full_code + "\n```\n\n</details>"
notebook.metadata["lab_execution"] = {
    "source_sha256": summary["source_sha256"],
    "selection_rule": "minimum nested training CV RMSE; never minimum test RMSE",
}
nbformat.validate(notebook)
nbformat.write(notebook, path)
(ROOT / "lab1.py").write_text(
    '# -*- coding: utf-8 -*-\n"""Полный листинг лабораторной №1. Запускать из корня AIML."""\n\n'
    + full_code + "\n", encoding="utf-8"
)
exporter = HTMLExporter(template_name="lab", embed_images=True)
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
body, _ = exporter.from_notebook_node(notebook)
(ROOT / "report.html").write_text(body, encoding="utf-8")
print(f"Notebook, report.html and lab1.py saved; elapsed {time.monotonic() - started:.1f}s", flush=True)
