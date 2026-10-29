"""Check report completeness and recompute all test metrics from predictions."""
from pathlib import Path
import hashlib
import json
import re
import nbformat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.read(ROOT / "lab1.ipynb", as_version=4)
nbformat.validate(nb)
codes = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
assert [c.execution_count for _, c in codes] == list(range(1, len(codes) + 1))
images = 0
for i, cell in codes:
    assert nb.cells[i + 1].cell_type == "markdown" and len(nb.cells[i + 1].source) > 100
    for output in cell.outputs:
        assert output.output_type != "error"
        images += int("image/png" in output.get("data", {}))
assert images == len(list((ROOT / "results/figures").glob("*.png"))) == 6
text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
assert all(f"### {i}. " in text for i in range(1, 8))
full_code = "\n\n".join(c.source for _, c in codes)
assert (ROOT / "lab1.py").read_text(encoding="utf-8").endswith(full_code + "\n")

source = json.loads((ROOT / "data/source.json").read_text(encoding="utf-8"))
csv_path = ROOT / "data" / source["file"]
assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == source["sha256"]
raw = pd.read_csv(csv_path).dropna().drop_duplicates()
train_ids, test_ids = train_test_split(raw.index, test_size=.20, random_state=42)
assert not set(train_ids).intersection(test_ids)
pred = pd.read_csv(ROOT / "results/predictions.csv").set_index("row_id")
assert pred.index.is_unique and set(pred.index) == set(test_ids)
truth = raw.loc[pred.index, "Appliances"].to_numpy()
assert np.array_equal(pred.y_true.to_numpy(), truth)
metrics = pd.read_csv(ROOT / "results/metrics.csv")
assert set(zip(metrics["Признаки"], metrics["Модель"])) == {
    (basis, model) for basis in ["Исходные", "PCA"] for model in ["OLS", "Lasso", "Ridge"]}
for _, row in metrics.iterrows():
    basis = "original" if row["Признаки"] == "Исходные" else "pca"
    error = truth - pred[f"{basis}_{row['Модель']}"].to_numpy()
    independent = {"RMSE, Wh": np.sqrt(np.mean(error ** 2)),
                   "R²": 1 - np.sum(error ** 2) / np.sum((truth - truth.mean()) ** 2),
                   "MAPE, %": 100 * np.mean(np.abs(error / truth))}
    assert all(np.isclose(row[key], value, rtol=1e-10) for key, value in independent.items())

features = raw.drop(columns=["date", "Appliances", "rv1", "rv2"])
spectrum = PCA().fit(StandardScaler().fit_transform(features.loc[train_ids])).explained_variance_ratio_.cumsum()
k = int(metrics.loc[metrics["Признаки"].eq("PCA"), "k"].iloc[0])
assert spectrum[k - 2] < .95 <= spectrum[k - 1]
vif = pd.read_csv(ROOT / "results/vif.csv")
assert np.allclose(vif.loc[vif["Базис"].eq("PCA"), "VIF"], 1, atol=1e-8)
html = (ROOT / "report.html").read_text(encoding="utf-8")
assert len(re.findall(r'<img\b[^>]*src="data:image/png;base64,', html)) == images
print(f"PASS: {len(codes)} executed cells with Markdown explanations; 7 sections; {images} plots.")
print("PASS: all 18 test metrics match predictions; dataset, split and PCA/VIF checks pass.")
