"""Check executed explanations and independently recompute reported test metrics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
nb = nbformat.read(ROOT / "lab1.ipynb", as_version=4)
nbformat.validate(nb)
code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
counts = []
image_count = 0
for i, cell in code_cells:
    assert cell.execution_count is not None, f"Unexecuted cell {i + 1}"
    counts.append(cell.execution_count)
    explanation = nb.cells[i + 1]
    assert explanation.cell_type == "markdown" and len(explanation.source.strip()) > 100
    for output in cell.outputs:
        assert output.output_type != "error", f"Error in cell {i + 1}"
        image_count += int("image/png" in output.get("data", {}))
assert counts == list(range(1, len(code_cells) + 1)), counts
assert image_count == 11, f"Expected 11 plot outputs, got {image_count}"

metadata = json.loads((ROOT / "data/source.json").read_text(encoding="utf-8"))
checksum = hashlib.sha256((ROOT / "data" / metadata["file"]).read_bytes()).hexdigest()
assert checksum == metadata["sha256"]
raw = pd.read_csv(ROOT / "data" / metadata["file"])
assert raw.date.is_monotonic_increasing
split = pd.read_csv(ROOT / "results/split_random.csv")
assert split.row_id.is_unique and len(split) == len(raw)
test_ids = set(split.loc[split.split.eq("test"), "row_id"])
train_ids = set(split.loc[split.split.eq("train"), "row_id"])
assert not test_ids.intersection(train_ids)

predictions = pd.read_csv(ROOT / "results/predictions_random_test.csv").set_index("row_id")
assert predictions.index.is_unique and set(predictions.index) == test_ids
assert np.array_equal(predictions.Appliances.to_numpy(), raw.loc[predictions.index, "Appliances"].to_numpy())
reported = pd.read_csv(ROOT / "results/metrics_all.csv")
assert set(zip(reported.Features, reported.Model)) == {
    (feature_set, model) for feature_set in ["Original", "PCA95"] for model in ["OLS", "Lasso", "Ridge"]
}
truth = predictions.Appliances.to_numpy()
for _, row in reported.iterrows():
    prediction = predictions[f"{row.Features}_{row.Model}"].to_numpy()
    residual = truth - prediction
    independent_metrics = {
        "Test_RMSE": np.sqrt(np.mean(residual ** 2)),
        "Test_R2": 1 - np.sum(residual ** 2) / np.sum((truth - truth.mean()) ** 2),
        "Test_MAPE_pct": 100 * np.mean(np.abs(residual / truth)),
    }
    for metric, actual in independent_metrics.items():
        assert np.isclose(row[metric], actual, rtol=1e-10, atol=1e-10), (row.Model, metric)
    assert all(np.isfinite(row[k]) for k in row.index if k not in ["Features", "Model", "alpha"])

vif = pd.read_csv(ROOT / "results/vif_pca.csv")
assert np.allclose(vif.VIF, 1, atol=1e-8)
spectrum = pd.read_csv(ROOT / "results/pca_variance.csv")
summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
k = summary["n_components"]
assert spectrum.iloc[k - 1].cumulative_ratio >= .95
assert spectrum.iloc[k - 2].cumulative_ratio < .95
selected = reported.loc[reported.CV_RMSE_mean.idxmin()]
assert summary["selected_by_nested_cv"] == {"Model": selected.Model, "Features": selected.Features}
html = (ROOT / "report.html").read_text(encoding="utf-8")
assert html.count("data:image/png;base64,") >= 11
assert "Развернуть полный листинг Python" in html
assert "Найден минимум среди проверенных" in html
print(f"PASS: {len(code_cells)} executed cells, {image_count} plots, all Markdown explanations present.")
print("PASS: all 18 test metrics independently match saved predictions; data, split and PCA checks pass.")
