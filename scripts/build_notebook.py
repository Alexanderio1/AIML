"""Build the self-contained Russian lab notebook. Run run_notebook.py afterward."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(source, explanation, key=None):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))
    cell = nbf.v4.new_markdown_cell(textwrap.dedent(explanation).strip())
    if key:
        cell.metadata["result_key"] = key
        cell.metadata["base_source"] = cell.source
    cells.append(cell)


md(r"""
# Лабораторная работа № 1
## Линейная регрессия и факторный анализ

**Датасет:** Appliances Energy Prediction.  
**Целевая переменная:** `Appliances`, расход электроэнергии бытовыми приборами, Wh (Вт·ч).  
**Формат:** выполненный вычислительный отчёт. Автор, группа и преподаватель заполняются студентом.

### 1. Введение

Цель работы — изучить линейную регрессию, обучить модели на реальных измерениях,
оценить их качество и исследовать влияние мультиколлинеарности и PCA.
Решаются задачи загрузки и анализа данных, предобработки, расчёта VIF, обучения
OLS, Lasso и Ridge, диагностики остатков и повторного обучения на главных компонентах.

`Appliances` — количественная, хотя и округлённая величина, поэтому используется регрессия,
а не классификация или логистическая регрессия. Сравниваются шесть моделей:
три на исходных подготовленных признаках и три после PCA.

**План оценки.** Основное разбиение — случайное 80/20 с `random_state=42`.
На обучающей части используется вложенная кросс-валидация: 5 внешних фолдов
для оценки и 3 внутренних для настройки `alpha`. Число компонент задаётся
правилом сохранения не менее 95% дисперсии, а не подбирается по тесту.
Все обучаемые преобразования входят в `Pipeline` и заново оцениваются внутри
каждого фолда. Итоговый тест не участвует в настройке.

Наблюдения образуют временной ряд одного дома. Случайный тест характеризует
восстановление пропущенных интервалов при известных одновременных признаках,
а не строгий прогноз будущего. В конце приведена отдельная проверка по времени.

### Обозначения и метрики

Для $n$ наблюдений, истинных значений $y_i$ и прогнозов $\hat y_i$:

$$RMSE=\sqrt{\frac1n\sum_{i=1}^n(y_i-\hat y_i)^2},\qquad
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},$$
$$MAPE=\frac{100\%}{n}\sum_{i=1}^n\left|\frac{y_i-\hat y_i}{y_i}\right|.$$

RMSE измеряется в Wh, $R^2$ безразмерен, MAPE приводится в процентах.
Для RMSE и MAPE меньше — лучше; для $R^2$ больше — лучше.
При малых $y_i$ MAPE сильно увеличивается, а при нуле не определён в такой записи.
Поэтому наличие нулей проверяется, а основной критерий выбора — RMSE.
Отрицательный $R^2$ допустим: модель хуже постоянного прогноза среднего по этому тесту.
Прогнозы линейных моделей не обрезаются до нуля, чтобы не изменять сравниваемые методы.
""")

code(r"""
from pathlib import Path
import os
import sys
import json
import hashlib
import warnings
from importlib.metadata import version

ROOT = Path.cwd()
if not (ROOT / "data/source.json").exists():
    raise RuntimeError("Запускайте notebook или lab1.py из корня проекта AIML.")
for name, directory in {
    "MPLCONFIGDIR": ROOT / ".cache/matplotlib",
    "XDG_CACHE_HOME": ROOT / ".cache",
}.items():
    os.environ.setdefault(name, str(directory))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
from IPython import get_ipython
from IPython.display import display
if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (root_mean_squared_error, r2_score,
                             mean_absolute_percentage_error, make_scorer)
from sklearn.model_selection import (train_test_split, KFold, TimeSeriesSplit,
                                     GridSearchCV, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
OUT = ROOT / "results"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
notes = {}
thread_limiter = threadpool_limits(limits=1)
warnings.filterwarnings("error", category=ConvergenceWarning)
pd.set_option("display.max_columns", 16)
pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format", "{:.4f}".format)
sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160})

def show_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, bbox_inches="tight")
    plt.show()
    plt.close()

packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
            "seaborn", "statsmodels", "nbformat", "nbclient", "nbconvert", "ipykernel"]
versions = {package: version(package) for package in packages}
display(pd.Series(versions, name="Версия").to_frame())
""", """
Зафиксированы версии библиотек и начальное состояние генератора случайных чисел.
Параллелизм BLAS ограничен одним потоком: для небольшого числа признаков это
уменьшает накладные расходы. Предупреждение о несходимости Lasso считается ошибкой,
а не скрывается. Все таблицы и рисунки сохраняются в `results/`.
""")

md("""
### 2. Описание датасета и проверка загрузки

Выбран [Appliances Energy Prediction на Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction).
Его первоисточник — измерения одного энергоэффективного дома в Бельгии,
опубликованные Candanedo, Feldheim и Deramaix в 2017 году. Данные собирались
с января по май 2016 года с шагом 10 минут. Часть погодных показателей
взята с метеостанции Chièvres и интерполирована до этого шага.

В этой работе CSV получен из [репозитория автора исследования](https://github.com/LuisM78/Appliances-energy-prediction-data),
а не через API Kaggle. Это исходный набор того же исследования; побайтовое
совпадение с конкретной версией архива Kaggle отдельно не проверялось.
Адрес привязан к коммиту, контрольная сумма записана в `data/source.json`.
Локальная копия включена в проект: повторный запуск не требует сети и токена Kaggle.

| Столбцы | Содержание и единицы |
|---|---|
| `date` | Время измерения, без явно заданного часового пояса |
| `Appliances` | Энергия бытовых приборов за интервал, Wh; целевая переменная |
| `lights` | Энергия освещения, Wh |
| `T1`–`T9` | Температура, °C: кухня, гостиная, прачечная, кабинет, ванная, северная наружная сторона, гладильная, детская, спальня родителей |
| `RH_1`–`RH_9` | Относительная влажность в соответствующих местах, % |
| `T_out`, `RH_out` | Наружная температура (°C) и влажность (%) по метеостанции |
| `Press_mm_hg` | Атмосферное давление, мм рт. ст. |
| `Windspeed`, `Visibility` | Скорость ветра, м/с; видимость, км |
| `Tdewpoint` | Температура точки росы, °C |
| `rv1`, `rv2` | Искусственные безразмерные случайные контрольные переменные |

Не следует называть `Appliances` мощностью в ваттах: в описании источника указана
энергия в Wh. Данные одного дома также не позволяют автоматически переносить выводы
на все здания или все времена года.
""")

code(r"""
source = json.loads((ROOT / "data/source.json").read_text(encoding="utf-8"))
csv_path = ROOT / "data" / source["file"]
actual_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
if actual_hash != source["sha256"]:
    raise ValueError("Контрольная сумма CSV отличается от зафиксированной в source.json.")
raw = pd.read_csv(csv_path)
assert raw.shape == (source["rows"], source["columns"])
numeric_columns = raw.columns.drop("date")
raw[numeric_columns] = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw[numeric_columns] = raw[numeric_columns].replace([np.inf, -np.inf], np.nan)
audit = pd.Series({
    "Строк": len(raw), "Столбцов": raw.shape[1],
    "Пропущенных ячеек после приведения типов": int(raw.isna().sum().sum()),
    "Повторяющихся строк": int(raw.duplicated().sum()),
    "Повторяющихся временных меток": int(raw["date"].duplicated().sum()),
    "Нулевых значений цели": int(raw["Appliances"].eq(0).sum()),
    "Минимальная цель, Wh": raw["Appliances"].min(),
    "Максимальная цель, Wh": raw["Appliances"].max(),
})
display(raw.head())
display(audit.to_frame("Значение"))
display(raw.describe().T)
display(raw["date"].sort_values().diff().value_counts().to_frame("Интервалов"))
notes["audit"] = (
    f"Загружено **{len(raw):,} строк и {raw.shape[1]} столбцов**. "
    f"Период: **{raw.date.min()} — {raw.date.max()}**. "
    f"Пропущенных ячеек — **{raw.isna().sum().sum()}**, дубликатов строк — "
    f"**{raw.duplicated().sum()}**. Цель принимает значения от "
    f"**{raw.Appliances.min():.0f} до {raw.Appliances.max():.0f} Wh**, "
    f"её медиана — **{raw.Appliances.median():.0f} Wh**, "
    f"среднее — **{raw.Appliances.mean():.2f} Wh**. "
    "Среднее заметно выше медианы: распределение имеет правый хвост. "
    "Нулей в цели нет, поэтому MAPE вычисляется без искусственного epsilon."
)
""", """
Контрольная сумма проверяет, что анализируется именно зафиксированный файл.
Проверка интервалов нужна для последующего анализа временной зависимости.
Представленная сводка всего набора служит описанием данных; модель и её параметры
будут обучаться только на тренировочной части.
""", "audit")

md("""
### 3. Подготовка данных

Пропуски удаляются целыми строками (`dropna`), как требует задание;
если их нет, операция не меняет набор. Заполнение средним или медианой не применяется.
Полные дубликаты удаляются. Реальные пики потребления не объявляются ошибками
только из-за большого значения: они важны для оценки RMSE.

`date` не подаётся в модель как строка или номер эпохи. Вместо этого
создаются четыре циклических признака: синус и косинус времени суток
и дня недели. Например, 23:50 и 00:00 остаются близкими на окружности.
Эти преобразования детерминированы и не оценивают параметры по данным.
`rv1` и `rv2` исключаются по описанию источника, а не по результатам теста:
это искусственные случайные контроли. Категориальных признаков после этого нет,
поэтому One-Hot Encoding не требуется.

Сначала выполняется разбиение; `StandardScaler` обучается позднее,
только внутри `Pipeline`. Цель не стандартизируется и не логарифмируется:
все шесть моделей сравниваются на одной шкале Wh.
""")

code(r"""
data = raw.dropna().drop_duplicates().sort_values("date").reset_index(drop=True)
assert data["date"].is_unique
assert data["Appliances"].gt(0).all(), "Для выбранной формулы MAPE цель должна быть > 0."
assert np.array_equal(data["rv1"].to_numpy(), data["rv2"].to_numpy())
X = data.drop(columns=["date", "Appliances", "rv1", "rv2"]).copy()
hour = data["date"].dt.hour + data["date"].dt.minute / 60
weekday = data["date"].dt.dayofweek
X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
X["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
X["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
y = data["Appliances"].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, shuffle=True
)
assert set(X_train.index).isdisjoint(X_test.index)
assert X_train.shape[1] == 29
pd.DataFrame({"row_id": data.index,
              "split": np.where(data.index.isin(X_train.index), "train", "test")}
             ).to_csv(OUT / "split_random.csv", index=False)
display(pd.DataFrame({"Выборка": ["Train", "Test"],
                      "Строк": [len(X_train), len(X_test)],
                      "Признаков": [X_train.shape[1], X_test.shape[1]]}))
notes["preparation"] = (
    f"После очистки осталось **{len(data):,}** наблюдений; удалено "
    f"**{len(raw) - len(data)}** строк. Train содержит **{len(X_train):,}**, "
    f"test — **{len(X_test):,}** наблюдений. Используются **25 физических "
    "предикторов + 4 календарных**, всего **29**. "
    "Проверка подтвердила точное равенство `rv1 == rv2`: при их совместном "
    "включении VIF этих столбцов был бы бесконечным. Удаление случайных "
    "контролей не устраняет корреляции между реальными датчиками."
)
""", """
Разбиение сохранено по индексам строк, поэтому все модели получают одинаковые
train и test. Исходная дата остаётся в `data` для проверки по времени,
но отсутствует в матрице регрессоров. Освещение и погодные измерения здесь
считаются известными одновременно с оцениваемым потреблением; для прогноза
заранее их доступность потребовала бы отдельного обоснования.
""", "preparation")

md("""
### 4. Ход работы
#### 4.1. Распределения и зависимости на обучающей выборке

Начиная с этого раздела, исследовательские графики и VIF строятся на **train**.
Распределения теста не используются для выбора признаков, настройки моделей
или решения о числе компонент.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(y_train, bins=60, ax=axes[0], color="#176b87")
axes[0].set(title="Распределение Appliances", xlabel="Appliances, Wh", ylabel="Число наблюдений")
sns.boxplot(x=y_train, ax=axes[1], color="#64b5a0")
axes[1].set(title="Диаграмма размаха", xlabel="Appliances, Wh")
sns.histplot(np.log1p(y_train), bins=50, ax=axes[2], color="#d18f42")
axes[2].set(title="Логарифмическая шкала (только EDA)",
            xlabel="ln(1 + Appliances)", ylabel="Число наблюдений")
show_figure("01_target_distribution.png")
notes["target"] = (
    f"На train асимметрия цели равна **{y_train.skew():.2f}**, "
    f"95-й процентиль — **{y_train.quantile(.95):.0f} Wh**. "
    "Редкие большие значения дают существенный вклад в квадратичную ошибку. "
    "Много наблюдений сосредоточено на низком потреблении, поэтому высокая "
    "MAPE возможна даже при умеренной абсолютной ошибке. "
    "Логарифм на третьем графике нужен только для изучения формы распределения; "
    "обучение далее выполняется на исходной цели."
)
""", """
Диаграмма размаха выделяет экстремальные по статистическому правилу значения,
но не доказывает, что они ошибочны. Удаление этих измерений без предметного
обоснования искусственно облегчило бы задачу и изменило объект исследования.
""", "target")

code(r"""
fig, axes = plt.subplots(6, 5, figsize=(18, 19))
for ax, column in zip(axes.flat, X_train.columns):
    ax.hist(X_train[column], bins=35, color="#3a829e", alpha=.85)
    ax.set(title=column, ylabel="Частота")
for ax in axes.flat[X_train.shape[1]:]:
    ax.axis("off")
fig.suptitle("Распределения всех 29 подготовленных признаков на train", y=1.01, fontsize=17)
show_figure("02_predictor_distributions.png")
""", """
Температуры измеряются десятками градусов, давление — сотнями мм рт. ст.,
а циклические признаки лежат в диапазоне от −1 до 1. Поэтому единый штраф
Lasso или Ridge без стандартизации был бы несопоставим для разных столбцов.
Масса нулей у `lights` описывает интервалы без расхода на освещение, а не пропуски.
Ненормальность распределений признаков сама по себе не запрещает линейную регрессию.
""")

code(r"""
train_eda = X_train.assign(Appliances=y_train)
correlation = train_eda.corr(method="pearson")
correlation.to_csv(OUT / "correlation_train.csv")
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(correlation, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            square=True, ax=ax, cbar_kws={"label": "Корреляция Пирсона"})
ax.set_title("Матрица корреляций: train, признаки и целевая переменная")
show_figure("03_correlation_matrix.png")
target_corr = correlation["Appliances"].drop("Appliances").sort_values(
    key=lambda s: s.abs(), ascending=False
)
display(target_corr.to_frame("r с Appliances"))
top_features = target_corr.head(6).index
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
sample_index = X_train.sample(n=min(3000, len(X_train)), random_state=SEED).index
for ax, column in zip(axes.flat, top_features):
    sns.regplot(x=X_train.loc[sample_index, column], y=y_train.loc[sample_index],
                ci=None, scatter_kws={"s": 8, "alpha": .17},
                line_kws={"color": "#b53434"}, ax=ax)
    ax.set(title=f"{column}: r = {target_corr[column]:.3f}", ylabel="Appliances, Wh")
show_figure("04_target_relationships.png")
notes["correlation"] = (
    f"Максимальная по модулю парная связь с целью: **{target_corr.index[0]}**, "
    f"**r = {target_corr.iloc[0]:.3f}**. Даже эта связь далека от ±1: "
    "простая линейная зависимость от одного датчика не описывает большинство "
    "колебаний потребления. На матрице заметны блоки связанных температур "
    "и влажностей. Это основание проверить VIF, но сама парная корреляция "
    "не измеряет все многомерные линейные зависимости и не доказывает причинность."
)
""", """
Корреляции вычислены по всему train; на точечных графиках показана только
воспроизводимая подвыборка из 3000 точек для читаемости. Красная прямая
иллюстрирует одномерный тренд и не является итоговой многомерной моделью.
""", "correlation")

md(r"""
#### 4.2. Мультиколлинеарность и VIF

Для каждого предиктора $x_j$ строится вспомогательная регрессия на остальные
предикторы со свободным членом. Тогда
$$VIF_j=\frac{1}{1-R_j^2}.$$
Значения выше 5 рассматриваются как предупреждение, выше 10 — как сильная
мультиколлинеарность; это эвристические пороги. Целевая переменная
в расчёт VIF **не включается**. Стандартизация не устраняет VIF:
масштаб признака не меняет его линейную зависимость от других.
""")

code(r"""
def vif_table(frame):
    a = StandardScaler().fit_transform(frame)
    records = []
    for j, column in enumerate(frame.columns):
        others = np.delete(a, j, axis=1)
        fitted = LinearRegression().fit(others, a[:, j]).predict(others)
        rss = np.square(a[:, j] - fitted).sum()
        tss = np.square(a[:, j] - a[:, j].mean()).sum()
        value = np.inf if rss <= 1e-12 * tss else tss / rss
        records.append({"Признак": column, "VIF": value})
    return pd.DataFrame(records).set_index("Признак").sort_values("VIF", ascending=False)

vif_original = vif_table(X_train)
vif_original.to_csv(OUT / "vif_original.csv")
display(vif_original)
fig, ax = plt.subplots(figsize=(10, 8))
vif_original.sort_values("VIF").plot.barh(y="VIF", legend=False, color="#3a829e", ax=ax)
ax.set_xscale("log")
ax.axvline(5, color="#e6a23c", linestyle="--", label="VIF = 5")
ax.axvline(10, color="#b53434", linestyle="--", label="VIF = 10")
ax.set(title="VIF исходных подготовленных признаков (train)", xlabel="VIF, логарифмическая шкала")
ax.legend()
show_figure("05_vif_original.png")
notes["vif"] = (
    f"**{int(vif_original.VIF.gt(10).sum())} из {len(vif_original)}** признаков "
    f"имеют VIF > 10; максимальный VIF у **{vif_original.index[0]}** "
    f"равен **{vif_original.VIF.iloc[0]:.2f}**. "
    "Значит, даже после исключения случайных дублей присутствует сильная "
    "мультиколлинеарность. Она повышает нестабильность отдельных коэффициентов "
    "OLS, но не обязана делать его прогноз хуже на похожих данных. "
    "Ridge и Lasso ограничивают коэффициенты штрафом; PCA заменяет "
    "коррелированные столбцы ортогональными направлениями."
)
""", """
Вспомогательные регрессии центрируются со свободным членом; константа не
публикуется как отдельный «признак с VIF». Бесконечность для точной зависимости
обрабатывается явно. Логарифмическая ось позволяет одновременно видеть
умеренные и очень большие значения VIF.
""", "vif")

md(r"""
#### 4.3. Модели на исходных подготовленных признаках

OLS минимизирует $\frac{1}{2n}\|y-X\beta\|_2^2$.
Lasso добавляет $\alpha\|\beta\|_1$ и может занулять коэффициенты.
Ridge в реализации scikit-learn минимизирует
$\|y-X\beta\|_2^2+\alpha\|\beta\|_2^2$ и плавно сжимает коэффициенты.
Из-за разных определений масштаба штрафа численные `alpha` этих методов
не сравниваются между собой. Свободный член не штрафуется.

Для Lasso проверяются 9 значений `alpha` от 0.01 до 100,
для Ridge — 17 значений от 0.001 до 100000, по логарифмической сетке.
Критерий выбора — минимальный RMSE внутренней CV.
Внешние фолды одни и те же для всех шести моделей. После вложенной оценки
подбор повторяется на полном train, затем один раз вычисляются тестовые метрики.
Стандартное отклонение по фолдам — разброс оценок, **не доверительный интервал**.
""")

code(r"""
SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
    "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
}
outer_folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train))
inner_folds = KFold(n_splits=3, shuffle=True, random_state=SEED + 1)
alphas = {"Lasso": np.logspace(-2, 2, 9), "Ridge": np.logspace(-3, 5, 17)}
REGRESSORS = {
    "OLS": LinearRegression(),
    "Lasso": Lasso(max_iter=50000, tol=1e-4, selection="cyclic"),
    "Ridge": Ridge(solver="svd"),
}

def model_pipeline(name, use_pca=False):
    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=.95, svd_solver="full")))
    steps.append(("reg", clone(REGRESSORS[name])))
    return Pipeline(steps)

def tuned_model(name, use_pca, cv):
    pipe = model_pipeline(name, use_pca)
    if name == "OLS":
        return pipe
    return GridSearchCV(pipe, {"reg__alpha": alphas[name]}, scoring="neg_root_mean_squared_error",
                        cv=cv, n_jobs=1, refit=True, error_score="raise")

def metrics(truth, prediction):
    return {"RMSE": float(root_mean_squared_error(truth, prediction)),
            "R2": float(r2_score(truth, prediction)),
            "MAPE_pct": float(100 * mean_absolute_percentage_error(truth, prediction))}

def train_family(use_pca):
    models, rows = {}, []
    for name in REGRESSORS:
        estimator = tuned_model(name, use_pca, inner_folds)
        scores = cross_validate(estimator, X_train, y_train, cv=outer_folds,
                                scoring=SCORING, n_jobs=1, error_score="raise")
        estimator.fit(X_train, y_train)
        fitted = estimator.best_estimator_ if isinstance(estimator, GridSearchCV) else estimator
        models[name] = fitted
        row = {"Features": "PCA95" if use_pca else "Original", "Model": name,
               "alpha": estimator.best_params_["reg__alpha"] if name != "OLS" else np.nan,
               "n_features": fitted["pca"].n_components_ if use_pca else X_train.shape[1]}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            values = values if metric == "R2" else -values
            if metric == "MAPE":
                values = values * 100
            row[f"CV_{metric}_mean"] = float(values.mean())
            row[f"CV_{metric}_std"] = float(values.std(ddof=1))
        row.update({f"Train_{k}": v for k, v in metrics(y_train, fitted.predict(X_train)).items()})
        row.update({f"Test_{k}": v for k, v in metrics(y_test, fitted.predict(X_test)).items()})
        if isinstance(estimator, GridSearchCV):
            pd.DataFrame(estimator.cv_results_).to_csv(
                OUT / f"grid_{row['Features']}_{name}.csv", index=False
            )
        rows.append(row)
    return models, pd.DataFrame(rows)

original_models, original_results = train_family(use_pca=False)
original_results.to_csv(OUT / "metrics_original.csv", index=False)
display(original_results)
ridge_row = original_results.set_index("Model").loc["Ridge"]
lasso_row = original_results.set_index("Model").loc["Lasso"]
notes["original_models"] = (
    f"На полном train внутренней CV выбраны **alpha(Lasso) = {lasso_row.alpha:g}**, "
    f"**alpha(Ridge) = {ridge_row.alpha:g}**. "
    f"У Ridge внешний CV RMSE = **{ridge_row.CV_RMSE_mean:.2f} ± "
    f"{ridge_row.CV_RMSE_std:.2f} Wh**, тестовый RMSE = "
    f"**{ridge_row.Test_RMSE:.2f} Wh**, R² = **{ridge_row.Test_R2:.4f}**, "
    f"MAPE = **{ridge_row.Test_MAPE_pct:.2f}%**. "
    "Различие train и test оценивается по таблице, а не по ошибке обучения в одиночку."
)
""", """
Колонки `CV_*` получены на внешних фолдах вложенной проверки; подбор `alpha`
внутри них не использовал их контрольные наблюдения. `Test_*` относятся к
одному отложенному тесту. Нельзя выдавать лучший внутренний CV-результат
`GridSearchCV` за независимую оценку качества; здесь эти величины разделены.
""", "original_models")

code(r"""
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
baseline_metrics = metrics(y_test, baseline.predict(X_test))
display(pd.DataFrame([baseline_metrics], index=["Прогноз среднего train"]))
coefs = pd.DataFrame({name: model["reg"].coef_ for name, model in original_models.items()},
                     index=X_train.columns)
coefs.to_csv(OUT / "coefficients_standardized_original.csv")
display(coefs.loc[coefs.OLS.abs().sort_values(ascending=False).head(12).index])
zero_lasso = int(np.isclose(original_models["Lasso"]["reg"].coef_, 0, atol=1e-10).sum())
notes["baseline"] = (
    f"Постоянный прогноз среднего train даёт на тесте **RMSE = "
    f"{baseline_metrics['RMSE']:.2f} Wh**, **R² = {baseline_metrics['R2']:.4f}**, "
    f"**MAPE = {baseline_metrics['MAPE_pct']:.2f}%**. "
    "Это необходимая точка отсчёта: положительный R² ещё не означает высокую точность. "
    f"Lasso занулил **{zero_lasso} из {X_train.shape[1]}** коэффициентов. "
    "Таблица показывает коэффициенты на стандартизированной шкале признаков: "
    "они соответствуют изменению признака на одно его стандартное отклонение "
    "при фиксированных остальных. При высокой мультиколлинеарности их нельзя "
    "трактовать как независимые причинные эффекты."
)
""", """
Средний прогноз вычислен исключительно по train. В таблице коэффициентов
показаны 12 наибольших по модулю коэффициентов OLS, а не отобранная по тесту
подмодель. Полный набор коэффициентов сохранён отдельно.
""", "baseline")

md(r"""
#### 4.4. Остатки исходных моделей

Остаток определяется как $e_i = y_i - \hat y_i$.
При гомоскедастичности вертикальный разброс вокруг нуля примерно одинаков
при разных прогнозах. Веер, изменение разброса, изгиб или отдельные кластеры
указывают на возможную гетероскедастичность либо недостаточную спецификацию.
""")

code(r"""
def residual_plot(models, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        prediction = model.predict(X_test)
        residual = y_test.to_numpy() - prediction
        ax.scatter(prediction, residual, s=9, alpha=.23, color="#267b9d")
        ax.axhline(0, color="#b53434", linestyle="--", linewidth=1.5)
        ax.set(title=name, xlabel="Прогноз, Wh", ylabel="Остаток y − прогноз, Wh")
    fig.suptitle(title, y=1.04)
    show_figure(filename)

residual_plot(original_models, "Остатки на тесте: исходные признаки", "06_residuals_original.png")
original_residual = y_test - original_models["OLS"].predict(X_test)
notes["residuals_original"] = (
    f"У OLS средний тестовый остаток **{original_residual.mean():.2f} Wh**, "
    f"максимальный положительный — **{original_residual.max():.2f} Wh**. "
    "Большие положительные остатки означают недооценку пиков потребления. "
    "Облако асимметрично, а нижняя граница наклонена: при положительной цели "
    "остаток ограничен снизу величиной min(y) − прогноз. Поэтому одной "
    "геометрии облака недостаточно для вывода о постоянстве дисперсии. "
    "Ниже визуальный анализ дополнен сравнением разброса по группам "
    "прогнозов и тестом Бройша–Пагана для OLS."
)
""", """
Все точки относятся к одному и тому же отложенному тесту; горизонтальная
линия показывает нулевую ошибку. Регуляризация может стабилизировать коэффициенты,
но не обязана устранять асимметрию и пропущенные нелинейные зависимости.
""", "residuals_original")

md(r"""
#### 4.5. Стандартизация и PCA

Сначала $z_j=(x_j-\mu_j)/s_j$, где $\mu_j$ и $s_j$ оцениваются на train.
PCA находит ортогональные направления максимальной дисперсии стандартизированных
признаков. Сохраняется минимальное число компонент, объясняющих не менее 95%
суммарной дисперсии. Это правило установлено заранее и не использует $y$.

Важно различать **дисперсию признаков** и качество прогноза цели:
направление с небольшой дисперсией может быть полезным для $y$.
PCA без усечения при полном ранге не улучшает OLS само по себе:
поворот базиса сохраняет пространство линейных прогнозов.
Изменение качества здесь связано с усечением этого пространства.
""")

code(r"""
diagnostic_scaler = StandardScaler().fit(X_train)
Z_train = diagnostic_scaler.transform(X_train)
full_pca = PCA(svd_solver="full").fit(Z_train)
cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
k95 = int(np.searchsorted(cumulative_variance, .95) + 1)
pca_diagnostic = PCA(n_components=.95, svd_solver="full").fit(Z_train)
assert pca_diagnostic.n_components_ == k95
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
pc_numbers = np.arange(1, X_train.shape[1] + 1)
axes[0].plot(pc_numbers, full_pca.explained_variance_, "o-", color="#267b9d")
axes[0].axvline(k95, linestyle="--", color="#b53434", label=f"k = {k95}")
axes[0].set(title="График каменистой осыпи", xlabel="Номер компоненты", ylabel="Собственное значение")
axes[0].legend()
axes[1].plot(pc_numbers, 100 * cumulative_variance, "o-", color="#267b9d")
axes[1].axhline(95, linestyle="--", color="#b53434", label="95%")
axes[1].axvline(k95, linestyle="--", color="#666666")
axes[1].set(title="Накопленная объяснённая дисперсия", xlabel="Число компонент", ylabel="Дисперсия, %")
axes[1].legend()
show_figure("07_pca_scree.png")
pd.DataFrame({"PC": pc_numbers, "eigenvalue": full_pca.explained_variance_,
              "explained_ratio": full_pca.explained_variance_ratio_,
              "cumulative_ratio": cumulative_variance}).to_csv(OUT / "pca_variance.csv", index=False)
notes["pca_scree"] = (
    f"Правило 95% оставляет **{k95} из {X_train.shape[1]}** компонент, "
    f"которые объясняют **{100 * cumulative_variance[k95-1]:.2f}%** дисперсии "
    "стандартизированных признаков. Размерность уменьшилась на "
    f"**{100 * (1 - k95 / X_train.shape[1]):.1f}%**. "
    "На левом графике показан спектр собственных значений, на правом — "
    "доля накопленной дисперсии; критерий выбора основан на правом графике, "
    "а не на субъективном поиске изгиба. Внутри CV PCA обучается заново, "
    "поэтому число компонент отдельного фолда может отличаться."
)
""", """
Этот PCA обучен только для диагностики на полном train. В оценке качества
не используется заранее преобразованный полный массив: преобразователь
входит в каждый оцениваемый `Pipeline`, предотвращая утечку между фолдами.
""", "pca_scree")

code(r"""
PC_train = pd.DataFrame(pca_diagnostic.transform(Z_train), index=X_train.index,
                        columns=[f"PC{i+1}" for i in range(k95)])
vif_pca = vif_table(PC_train)
vif_pca.to_csv(OUT / "vif_pca.csv")
display(vif_pca)
pc_correlation = PC_train.corr().to_numpy()
max_off_diagonal = float(np.max(np.abs(pc_correlation - np.eye(k95))))
assert np.allclose(vif_pca.VIF, 1.0, atol=1e-8)
directions = pd.DataFrame(pca_diagnostic.components_.T, index=X_train.columns,
                          columns=PC_train.columns)
directions.to_csv(OUT / "pca_directions.csv")
top_directions = []
for component in ["PC1", "PC2"]:
    top = directions[component].abs().nlargest(6).index
    for feature in top:
        top_directions.append({"Компонента": component, "Признак": feature,
                               "Вес в направлении": directions.loc[feature, component]})
display(pd.DataFrame(top_directions))
notes["pca_vif"] = (
    f"VIF всех сохранённых компонент находится в диапазоне "
    f"**{vif_pca.VIF.min():.8f}–{vif_pca.VIF.max():.8f}**; "
    f"максимальная абсолютная внедиагональная корреляция — **{max_off_diagonal:.2e}**. "
    "Таким образом, на обучающей выборке линейная мультиколлинеарность "
    "устранена с точностью вычислений. Ортогональность обеспечивается "
    "на данных обучения PCA; на новой выборке точная некоррелированность "
    "не гарантируется. Веса показывают, какие измерения образуют направления, "
    "но компоненты не являются автоматически физическими или причинными факторами."
)
""", """
В таблице весов приведены элементы собственных векторов, а не корреляционные
нагрузки. Знак собственного вектора можно целиком сменить без изменения
содержания PCA. PCA решает задачу ортогонального представления дисперсии;
это не тождественно статистической модели общих латентных факторов.
""", "pca_vif")

md("""
#### 4.6. Регрессии на главных компонентах и сравнение шести моделей

Повторяются те же три регрессии, сетки `alpha`, train/test и внешние фолды.
Единственное изменение конвейера — добавление `PCA(0.95)` после `StandardScaler`.
`whiten=False` (значение по умолчанию): после PCA компоненты не приводятся
к единичной дисперсии. Новые `alpha` подбираются отдельно в этом базисе.
""")

code(r"""
pca_models, pca_results = train_family(use_pca=True)
results = pd.concat([original_results, pca_results], ignore_index=True)
results = results.sort_values("CV_RMSE_mean").reset_index(drop=True)
results.to_csv(OUT / "metrics_all.csv", index=False)
display(results[["Features", "Model", "alpha", "n_features", "CV_RMSE_mean", "CV_RMSE_std",
                 "CV_R2_mean", "CV_R2_std", "CV_MAPE_mean", "CV_MAPE_std",
                 "Test_RMSE", "Test_R2", "Test_MAPE_pct"]])
before = original_results.set_index("Model")
after = pca_results.set_index("Model")
delta = pd.DataFrame({
    "ΔRMSE, Wh": after.Test_RMSE - before.Test_RMSE,
    "ΔR²": after.Test_R2 - before.Test_R2,
    "ΔMAPE, п.п.": after.Test_MAPE_pct - before.Test_MAPE_pct,
})
delta.to_csv(OUT / "metrics_pca_minus_original.csv")
display(delta)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for ax, metric, title, unit in zip(
    axes, ["Test_RMSE", "Test_R2", "Test_MAPE_pct"],
    ["RMSE — меньше лучше", "R² — больше лучше", "MAPE — меньше лучше"],
    ["Wh", "R²", "%"]
):
    pivot = results.pivot(index="Model", columns="Features", values=metric).reindex(["OLS", "Lasso", "Ridge"])
    pivot[["Original", "PCA95"]].plot.bar(ax=ax, color=["#267b9d", "#d18f42"], rot=0)
    baseline_key = {"Test_RMSE": "RMSE", "Test_R2": "R2", "Test_MAPE_pct": "MAPE_pct"}[metric]
    ax.axhline(baseline_metrics[baseline_key], linestyle="--", color="#666666", label="Среднее train")
    ax.set(title=title, xlabel="Модель", ylabel=unit)
    ax.legend(fontsize=8)
show_figure("08_model_comparison.png")
selected = results.iloc[0]
notes["comparison"] = (
    f"По минимальному **внешнему CV RMSE** выбрана конфигурация "
    f"**{selected['Model']} / {selected['Features']}**: "
    f"**{selected.CV_RMSE_mean:.2f} ± {selected.CV_RMSE_std:.2f} Wh**. "
    f"Её тестовые показатели: **RMSE = {selected.Test_RMSE:.2f} Wh**, "
    f"**R² = {selected.Test_R2:.4f}**, **MAPE = {selected.Test_MAPE_pct:.2f}%**. "
    "Выбор сделан по train CV, а не по минимальному тестовому числу. "
    f"Изменение тестового RMSE после PCA: OLS **{delta.loc['OLS', 'ΔRMSE, Wh']:+.2f}**, "
    f"Lasso **{delta.loc['Lasso', 'ΔRMSE, Wh']:+.2f}**, "
    f"Ridge **{delta.loc['Ridge', 'ΔRMSE, Wh']:+.2f} Wh**. "
    "В таблице Δ = после PCA минус до PCA: положительное ΔRMSE/ΔMAPE "
    "означает ухудшение, положительное ΔR² — улучшение. "
    "Близкие показатели не доказывают превосходство одной регуляризации "
    "без отдельного анализа неопределённости. "
    f"У Lasso/PCA выбрано alpha = {after.loc['Lasso', 'alpha']:g}; "
    "это нижняя граница проверенной сетки, то есть в данном базисе "
    "предпочтителен очень слабый штраф. Найден минимум среди проверенных "
    "значений, а не доказан глобальный оптимум alpha."
)
""", """
Таблица содержит независимую внешнюю CV для всех трёх метрик и результаты
единого теста. Выбор по RMSE может не совпасть с выбором по MAPE: эти метрики
по-разному взвешивают крупные пики и малые значения. PCA устраняет зависимости
предикторов, но не гарантирует улучшения прогноза.
""", "comparison")

md("""
#### 4.7. Остатки после PCA и проверка неоднородности разброса

Чтобы дополнить графики, тестовые наблюдения делятся на пять групп
по квантилям прогноза каждой модели. Сравниваются стандартные отклонения остатков.
Для **OLS**, обученных на train, дополнительно выполняется робастная к ненормальности
версия теста Бройша–Пагана (Koenker, `robust=True`). Нулевая гипотеза —
постоянство условной дисперсии ошибок; уровень значимости 0.05.
В качестве вспомогательных регрессоров берутся признаки соответствующего OLS
со свободным членом. Этот тест не применяется как формальное доказательство
для штрафных оценок Ridge и Lasso.
""")

code(r"""
residual_plot(pca_models, "Остатки на тесте: главные компоненты", "09_residuals_pca.png")
all_models = {("Original", name): model for name, model in original_models.items()}
all_models.update({("PCA95", name): model for name, model in pca_models.items()})
variance_rows = []
prediction_export = data.loc[y_test.index, ["date", "Appliances"]].copy()
for (feature_set, name), model in all_models.items():
    prediction = model.predict(X_test)
    prediction_export[f"{feature_set}_{name}"] = prediction
    r = pd.DataFrame({"prediction": prediction, "residual": y_test.to_numpy() - prediction})
    r["bin"] = pd.qcut(r.prediction, q=5, labels=False, duplicates="drop") + 1
    grouped = r.groupby("bin").agg(mean_prediction=("prediction", "mean"),
                                   residual_std=("residual", "std"), count=("residual", "size"))
    for bin_number, row in grouped.iterrows():
        variance_rows.append({"Features": feature_set, "Model": name, "bin": int(bin_number), **row.to_dict()})
variance_groups = pd.DataFrame(variance_rows)
variance_groups.to_csv(OUT / "residual_spread_bins.csv", index=False)
prediction_export.sort_values("date").to_csv(OUT / "predictions_random_test.csv", index_label="row_id")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, feature_set in zip(axes, ["Original", "PCA95"]):
    subset = variance_groups.query("Features == @feature_set")
    sns.lineplot(data=subset, x="bin", y="residual_std", hue="Model", marker="o", ax=ax)
    ax.set(title=feature_set, xlabel="Группа по квантилю прогноза (1 = низкие)",
           ylabel="Стандартное отклонение остатков, Wh", xticks=[1, 2, 3, 4, 5])
show_figure("10_residual_spread.png")
bp_rows = []
for feature_set, model in [("Original", original_models["OLS"]), ("PCA95", pca_models["OLS"])]:
    exog = sm.add_constant(model[:-1].transform(X_train), has_constant="add")
    train_residual = y_train.to_numpy() - model.predict(X_train)
    lm, p_value, f_value, f_p = het_breuschpagan(train_residual, exog, robust=True)
    bp_rows.append({"Features": feature_set, "LM": lm, "p_value": p_value,
                    "F": f_value, "F_p_value": f_p})
bp_results = pd.DataFrame(bp_rows)
bp_results.to_csv(OUT / "breusch_pagan_ols.csv", index=False)
display(bp_results.style.format({"p_value": "{:.3e}", "F_p_value": "{:.3e}"}))
ols_spread = variance_groups.query("Features == 'Original' and Model == 'OLS'").set_index("bin")
notes["heteroscedasticity"] = (
    f"У исходной OLS стандартное отклонение тестовых остатков растёт "
    f"с **{ols_spread.loc[1, 'residual_std']:.2f} Wh** в группе низких "
    f"прогнозов до **{ols_spread.loc[5, 'residual_std']:.2f} Wh** "
    "в группе высоких прогнозов. Это выраженная неоднородность разброса. "
    f"Для OLS исходных признаков p-value теста **{bp_results.iloc[0].p_value:.2e}**, "
    f"после PCA — **{bp_results.iloc[1].p_value:.2e}**. "
    + ("Оба значения ниже 0.05: при стандартных предпосылках нулевая гипотеза "
       "гомоскедастичности отвергается. " if bp_results.p_value.lt(.05).all()
       else "Результаты следует сопоставлять с уровнем 0.05 отдельно для каждого OLS. ")
    + "График группового разброса дополняет этот вывод наблюдаемым размером эффекта. "
    "PCA не исправляет модель условной дисперсии и не удаляет пики цели. "
    "Из-за временной зависимости наблюдений обычные p-value нельзя считать "
    "строгим доказательством: независимость ошибок здесь сомнительна. "
    "Это диагностическое свидетельство, а не проверка причинного механизма."
)
""", """
Проверка различий дисперсии не используется для повторного подбора моделей
по тесту. При необходимости статистических интервалов для OLS можно применять
HAC-ковариацию, но она изменит оценку неопределённости коэффициентов,
а не сами точечные прогнозы. Для улучшения прогнозов потребуется отдельно
исследовать спецификацию, например нелинейности и доступные временные лаги.
""", "heteroscedasticity")

md("""
#### 4.8. Дополнительная проверка временной структуры

Этот раздел дополняет, а не заменяет основное разбиение, требуемое в работе.
Первые 80% упорядоченных наблюдений отводятся для обучения, последние 20% — для теста
(`train_test_split(shuffle=False)`). Между ними оставляется разрыв в 144 наблюдения
(24 часа); эти строки не используются для обучения или оценки.
На временном train регуляризация заново настраивается с
`TimeSeriesSplit(n_splits=5, gap=144)`.
Значения тестовых метрик ниже не смешиваются с результатами случайного теста.

Такой эксперимент оценивает перенос на более поздний период этого же дома
при известных текущих признаках. Он включает сезонное изменение распределений;
разницу метрик нельзя целиком приписать только автокорреляции.
""")

code(r"""
X_time_train, X_time_test, y_time_train, y_time_test = train_test_split(
    X, y, test_size=.20, shuffle=False
)
GAP = 144
X_time_train = X_time_train.iloc[:-GAP]
y_time_train = y_time_train.iloc[:-GAP]
time_cv = TimeSeriesSplit(n_splits=5, gap=GAP)
time_rows, time_models = [], {}
for use_pca in [False, True]:
    for name in REGRESSORS:
        estimator = tuned_model(name, use_pca, time_cv)
        estimator.fit(X_time_train, y_time_train)
        fitted = estimator.best_estimator_ if isinstance(estimator, GridSearchCV) else estimator
        feature_set = "PCA95" if use_pca else "Original"
        time_models[(feature_set, name)] = fitted
        time_rows.append({"Features": feature_set, "Model": name,
                          "alpha": estimator.best_params_["reg__alpha"] if name != "OLS" else np.nan,
                          **metrics(y_time_test, fitted.predict(X_time_test))})
temporal_results = pd.DataFrame(time_rows)
time_baseline = DummyRegressor(strategy="mean").fit(X_time_train, y_time_train)
time_baseline_metrics = metrics(y_time_test, time_baseline.predict(X_time_test))
temporal_results.to_csv(OUT / "metrics_temporal.csv", index=False)
display(temporal_results)
display(pd.DataFrame([time_baseline_metrics], index=["Временной baseline"]))
selected_model = time_models[(selected["Features"], selected["Model"])]
time_prediction = selected_model.predict(X_time_test)
time_residual = y_time_test.to_numpy() - time_prediction
lag1 = float(pd.Series(time_residual).autocorr(lag=1))
dw = float(durbin_watson(time_residual))
fig, axes = plt.subplots(2, 1, figsize=(16, 7))
dates = data.loc[X_time_test.index, "date"]
axes[0].plot(dates, y_time_test, linewidth=.6, label="Факт", color="#267b9d")
axes[0].plot(dates, time_prediction, linewidth=.8, label="Прогноз", color="#d18f42")
axes[0].set(title="Последние 20% времени: конфигурация, выбранная по основной CV", ylabel="Энергия, Wh")
axes[0].legend()
axes[1].plot(dates, time_residual, linewidth=.6, color="#267b9d")
axes[1].axhline(0, color="#b53434", linestyle="--")
axes[1].set(xlabel="Дата", ylabel="Остаток, Wh", title="Остатки в порядке времени")
show_figure("11_temporal_check.png")
time_selected = temporal_results.query("Features == @selected.Features and Model == @selected.Model").iloc[0]
notes["temporal"] = (
    f"Временной train: **{len(X_time_train):,}** строк, тест: "
    f"**{len(X_time_test):,}**, разрыв: **{GAP}** строк. "
    f"Начало теста: **{dates.min()}**. Для конфигурации "
    f"**{selected['Model']} / {selected['Features']}** после перенастройки "
    f"на временном train получены **RMSE = {time_selected.RMSE:.2f} Wh**, "
    f"**R² = {time_selected.R2:.4f}**, **MAPE = {time_selected.MAPE_pct:.2f}%**. "
    f"Временной baseline: **RMSE = {time_baseline_metrics['RMSE']:.2f} Wh**. "
    f"Лаговая корреляция остатков **r₁ = {lag1:.3f}**, "
    f"статистика Дарбина–Уотсона **DW = {dw:.3f}**. "
    "DW здесь приведён как описательная диагностика, без табличного p-value. "
    "Заметная зависимость соседних ошибок ограничивает интерпретацию "
    "обычных случайных CV-оценок и тестов значимости."
)
""", """
На временном тесте нет подбора параметров; строки теста не включались
в обучение скейлера и PCA. Пики фактического потребления сравниваются
с линейным прогнозом в исходных единицах. Метеоданные и `lights`
одновременны с целью, поэтому этот эксперимент нельзя называть
готовой системой прогноза расхода за сутки вперёд.
""", "temporal")

md("""
### 5. Заключение

Выполнены загрузка и проверка происхождения данных, первичный анализ,
очистка, создание календарных признаков, расчёт корреляций и VIF,
обучение OLS, Lasso и Ridge с кросс-валидацией, диагностика остатков
и повторное обучение после стандартизированного PCA.
""")

code(r"""
notes["conclusion"] = (
    f"**1. Данные и зависимости.** Использованы {len(data):,} измерений одного дома. "
    f"Пропусков не обнаружено; удалены два искусственных случайных признака. "
    f"Среди {X_train.shape[1]} подготовленных признаков "
    f"{int(vif_original.VIF.gt(10).sum())} имеют VIF > 10.\n\n"
    f"**2. Качество исходного эксперимента.** Выбранная по внешней CV "
    f"конфигурация {selected['Model']} / {selected['Features']} имеет "
    f"тестовые RMSE = {selected.Test_RMSE:.2f} Wh, R² = {selected.Test_R2:.4f}, "
    f"MAPE = {selected.Test_MAPE_pct:.2f}%. "
    f"Baseline среднего: RMSE = {baseline_metrics['RMSE']:.2f} Wh. "
    "Ограниченный R² и большие ошибки на пиках показывают, что линейные "
    "зависимости не описывают основную вариативность поведения приборов.\n\n"
    f"**3. Эффект PCA.** Размерность снижена с {X_train.shape[1]} до {k95}, "
    "VIF компонент на train равен 1. "
    f"Изменение RMSE OLS после PCA составляет {delta.loc['OLS', 'ΔRMSE, Wh']:+.2f} Wh. "
    "Устранение мультиколлинеарности не равно автоматическому повышению "
    "точности: PCA сохраняет дисперсию X, а не обязательно информацию о y.\n\n"
    "**4. Остатки и ограничения.** Остатки асимметричны; пики потребления "
    "недооцениваются. Групповой разброс и тест Бройша–Пагана дают диагностику "
    "непостоянной дисперсии, но временная зависимость ограничивает формальную "
    "интерпретацию p-value. Дополнительный тест по времени показывает "
    "качество переноса на более поздние интервалы при известных признаках.\n\n"
    "**5. Возможные улучшения.** В отдельном эксперименте стоит проверить "
    "доступные на момент прогноза лаги и скользящие статистики, нелинейные "
    "взаимодействия, логарифмирование цели с корректным обратным преобразованием "
    "и нелинейные модели. Эти варианты здесь не обучались, поэтому улучшение "
    "метрик не утверждается. Для оценки обобщения нужны временные блоки "
    "и измерения других домов; результаты этой лабораторной не дают "
    "доказательства причинного влияния температуры или влажности на расход."
)
summary = {
    "notes": notes, "versions": versions, "seed": SEED,
    "rows": len(data), "n_train": len(X_train), "n_test": len(X_test),
    "n_features": X_train.shape[1], "n_components": k95,
    "pca_explained_variance": float(cumulative_variance[k95 - 1]),
    "baseline": baseline_metrics, "temporal_baseline": time_baseline_metrics,
    "selected_by_nested_cv": {"Model": selected["Model"], "Features": selected["Features"]},
    "source_sha256": actual_hash,
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
display(results[["Features", "Model", "Test_RMSE", "Test_R2", "Test_MAPE_pct"]])
""", """
Итоговая таблица повторяет метрики уже обученных моделей: повторного выбора
по тесту и скрытой перенастройки здесь нет. Выводы ниже относятся к
зафиксированным данным, разбиению и проверенным методам.
""", "conclusion")

md("""
### 6. Список источников

1. Candanedo L. M., Feldheim V., Deramaix D. **Data driven prediction models of energy use of appliances in a low-energy house.** Energy and Buildings. 2017. Vol. 140. P. 81–97. [DOI: 10.1016/j.enbuild.2017.01.083](https://doi.org/10.1016/j.enbuild.2017.01.083).
2. [Appliances Energy Prediction — Kaggle, loveall](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction). Выбранная пользователем публикация датасета.
3. [UCI Machine Learning Repository: Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction). Описание первоисточника. [DOI набора: 10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G).
4. [LuisM78/Appliances-energy-prediction-data](https://github.com/LuisM78/Appliances-energy-prediction-data/tree/e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1). Фактически использованный CSV, описание переменных и ссылка на статью; коммит `e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1`.
5. НГТУ НЭТИ. **Линейная регрессия и факторный анализ**. Учебные слайды, предоставленные с заданием: файл «Машинное обучение, регрессия и факторный анализ.pdf», слайды 33–43. Автор и год в доступном титуле не указаны.
6. Алетдинова А. А. **Методы и модели искусственного интеллекта в задачах нефтегазового комплекта** (название по титулу). НГТУ, кафедра автоматизированных систем управления. Материалы, предоставленные с заданием: файл «Машинное обучение.pdf». Год не указан.
7. scikit-learn: [линейные модели](https://scikit-learn.org/stable/modules/linear_model.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [вложенная CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [предотвращение утечки](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
8. statsmodels: [тест Бройша–Пагана / Koenker](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html).
9. [Dmewtriy/ML, lab1.ipynb](https://github.com/Dmewtriy/ML/blob/35827d46205abd70e2af8cde1e8fada48e8da600/lab1.ipynb). Указанный пользователем ориентир структуры; его данные и результаты в расчётах этой работы не использовались.

### 7. Приложение. Полный листинг программного кода

Все кодовые ячейки выше образуют самостоятельный воспроизводимый листинг,
вынесенный также в `lab1.py`. В HTML и notebook полный код дополнительно
собран ниже. Запуск: `python lab1.py` из корня проекта.
Каждая вычислительная ячейка notebook сопровождается отдельной
markdown-ячейкой с объяснением результата.
""")

appendix = nbf.v4.new_markdown_cell("Полный листинг будет добавлен после выполнения notebook.")
appendix.metadata["appendix_code"] = True
cells.append(appendix)
notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({
    "kernelspec": {"display_name": "Python (AIML)", "language": "python", "name": "aiml"},
    "language_info": {"name": "python", "version": "3.12"},
})
nbf.write(notebook, ROOT / "lab1.ipynb")
print(f"Created {len(cells)} cells in lab1.ipynb")
