# -*- coding: utf-8 -*-
"""Полный листинг лабораторной №1. Запускать из корня AIML."""

from pathlib import Path
import hashlib
import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from threadpoolctl import threadpool_limits

warnings.filterwarnings("error", category=ConvergenceWarning)
thread_limiter = threadpool_limits(limits=1)
sns.set_theme(style="whitegrid", font="DejaVu Sans")
pd.set_option("display.max_columns", 12)
OUT = Path("results")
(OUT / "figures").mkdir(parents=True, exist_ok=True)

def show_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / "figures" / name, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close()

source = json.loads(Path("data/source.json").read_text(encoding="utf-8"))
data_file = Path("data") / source["file"]
assert hashlib.sha256(data_file.read_bytes()).hexdigest() == source["sha256"]
raw = pd.read_csv(data_file, parse_dates=["date"])
display(raw.head(3))
display(pd.Series({"Строк": len(raw), "Столбцов": raw.shape[1],
                   "Пропусков": raw.isna().sum().sum(),
                   "Дубликатов строк": raw.duplicated().sum(),
                   "rv1 и rv2 совпадают": raw.rv1.equals(raw.rv2)}).to_frame("Значение"))
display(raw.Appliances.describe().to_frame("Appliances, Wh"))

numeric = raw.select_dtypes(include=np.number)
fig, axes = plt.subplots(6, 5, figsize=(17, 17))
for ax, column in zip(axes.flat, numeric.columns):
    ax.hist(numeric[column], bins=30, color="#3486a6")
    ax.set(title=column, ylabel="Частота")
for ax in axes.flat[len(numeric.columns):]:
    ax.axis("off")
fig.suptitle("Распределения числовых переменных, включая Appliances", y=1.01)
show_plot("01_distributions.png")

corr = numeric.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True)
plt.title("Корреляции Пирсона: исходный набор")
show_plot("02_correlations.png")
target_corr = corr["Appliances"].drop("Appliances").sort_values(key=abs, ascending=False)
display(target_corr.head(6).to_frame("Корреляция с Appliances"))

clean = raw.dropna().drop_duplicates().copy()
X = clean.drop(columns=["date", "Appliances", "rv1", "rv2"])
y = clean["Appliances"]
assert y.gt(0).all()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
display(pd.DataFrame({"Выборка": ["Train", "Test"],
                      "Строк": [len(X_train), len(X_test)], "Признаков": X.shape[1]}))

def vif(frame):
    matrix = add_constant(StandardScaler().fit_transform(frame))
    return pd.Series([variance_inflation_factor(matrix, i + 1) for i in range(frame.shape[1])],
                     index=frame.columns, name="VIF")

vif_before = vif(X_train)
display(vif_before.sort_values(ascending=False).round(2).to_frame())

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=17)
scoring = {"RMSE": "neg_root_mean_squared_error", "R2": "r2",
           "MAPE": "neg_mean_absolute_percentage_error"}

def assess_family(use_pca=False):
    candidates = {
        "OLS": (LinearRegression(), {}),
        "Lasso": (Lasso(max_iter=50000), {"reg__alpha": np.logspace(-2, 2, 9)}),
        "Ridge": (Ridge(solver="svd"), {"reg__alpha": np.logspace(-3, 5, 17)}),
    }
    rows, fitted = [], {}
    for name, (regressor, grid) in candidates.items():
        steps = [("scale", StandardScaler())]
        if use_pca:
            steps.append(("pca", PCA(n_components=.95, svd_solver="full")))
        pipe = Pipeline(steps + [("reg", regressor)])
        search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="neg_root_mean_squared_error",
                              n_jobs=1, error_score="raise")
        cv = cross_validate(search, X_train, y_train, cv=outer_cv, scoring=scoring,
                            n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        pred = fitted[name].predict(X_test)
        rows.append({
            "Признаки": "PCA" if use_pca else "Исходные", "Модель": name,
            "alpha": search.best_params_.get("reg__alpha", np.nan),
            "k": fitted[name]["pca"].n_components_ if use_pca else X.shape[1],
            "CV RMSE": -cv["test_RMSE"].mean(),
            "CV σ RMSE": cv["test_RMSE"].std(ddof=1),
            "CV R²": cv["test_R2"].mean(), "CV MAPE, %": -100 * cv["test_MAPE"].mean(),
            "RMSE, Wh": root_mean_squared_error(y_test, pred), "R²": r2_score(y_test, pred),
            "MAPE, %": 100 * mean_absolute_percentage_error(y_test, pred),
        })
    return pd.DataFrame(rows), fitted

base_metrics, base_models = assess_family()
display(base_metrics.round(4))

def residuals(models, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    spread = {}
    for ax, (name, model) in zip(axes, models.items()):
        pred = model.predict(X_test)
        error = y_test.to_numpy() - pred
        ax.scatter(pred, error, s=8, alpha=.22)
        ax.axhline(0, color="red", linestyle="--")
        ax.set(title=name, xlabel="Прогноз, Wh", ylabel="Остаток, Wh")
        groups = pd.Series(error).groupby(pd.qcut(pred, 5, labels=False)).std()
        spread[name] = [groups.iloc[0], groups.iloc[-1]]
    fig.suptitle(title, y=1.03)
    show_plot(filename)
    return pd.DataFrame(spread, index=["Низкие прогнозы", "Высокие прогнозы"]).T

spread_before = residuals(base_models, "Остатки: исходные признаки", "03_residuals_original.png")
display(spread_before.round(2))

scaled = StandardScaler().fit_transform(X_train)
pca_full = PCA(svd_solver="full").fit(scaled)
cumulative = pca_full.explained_variance_ratio_.cumsum()
k95 = int(np.searchsorted(cumulative, .95) + 1)
components = pd.DataFrame(pca_full.transform(scaled)[:, :k95],
                          columns=[f"PC{i+1}" for i in range(k95)])
vif_after = vif(components)
display(vif_after.round(6).to_frame())
pd.concat({"Исходные": vif_before, "PCA": vif_after}, names=["Базис", "Признак"]).to_csv(OUT / "vif.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
numbers = np.arange(1, X.shape[1] + 1)
axes[0].plot(numbers, pca_full.explained_variance_, "o-")
axes[0].axvline(k95, color="red", linestyle="--")
axes[0].set(title="Каменистая осыпь", xlabel="Номер компоненты", ylabel="Собственное значение")
axes[1].plot(numbers, 100 * cumulative, "o-")
axes[1].axhline(95, color="red", linestyle="--")
axes[1].axvline(k95, color="red", linestyle="--")
axes[1].set(title="Накопленная дисперсия", xlabel="Число компонент", ylabel="Доля дисперсии, %")
show_plot("04_pca_scree.png")

pca_metrics, pca_models = assess_family(use_pca=True)
comparison = pd.concat([base_metrics, pca_metrics], ignore_index=True)
comparison.to_csv(OUT / "metrics.csv", index=False)
display(comparison.round(4))
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, metric in zip(axes, ["RMSE, Wh", "R²", "MAPE, %"]):
    sns.barplot(data=comparison, x="Модель", y=metric, hue="Признаки", ax=ax)
    ax.set(xlabel="", title=metric)
    ax.get_legend().remove()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(.5, 1.08), ncol=2)
show_plot("05_model_comparison.png")
metric_names = ["RMSE, Wh", "R²", "MAPE, %"]
delta = (pca_metrics.set_index("Модель")[metric_names]
         - base_metrics.set_index("Модель")[metric_names])
display(delta.rename(columns=lambda s: "Δ " + s).round(4))
predictions = pd.DataFrame({"y_true": y_test})
for basis, models in [("original", base_models), ("pca", pca_models)]:
    for name, model in models.items():
        predictions[f"{basis}_{name}"] = model.predict(X_test)
predictions.to_csv(OUT / "predictions.csv", index_label="row_id")

spread_after = residuals(pca_models, "Остатки: главные компоненты", "06_residuals_pca.png")
display(spread_after.round(2))
