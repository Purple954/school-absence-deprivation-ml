# School Absence and Deprivation — ML Analysis

**[View Dashboard](https://Purple954.github.io/school-absence-deprivation-ml/dashboard.html)**

Persistent absence from school is one of the clearest early indicators of later educational disengagement, but the structural factors driving it are often treated as background noise rather than the main story. This project builds a machine learning pipeline on open DfE data to identify which school-level and area-level characteristics are most predictive of high absence rates — and uses explainability tools to make the model findings actionable.

---

## Research question

**Can neighbourhood deprivation, school characteristics, regional factors, and time predict persistent absenteeism rates in English state schools, and which features matter most?**

The analysis works at school level using the DfE's definition of persistent absence: pupils missing 10% or more of possible sessions. The target variable is the persistent absence rate (% of pupils persistently absent) per school. A four-year panel (2018-19, 2021-22, 2022-23, 2023-24) is used to capture both the pre-pandemic baseline and the post-lockdown trajectory. The 2019-20 and 2020-21 years are excluded as DfE did not publish school-level absence data during COVID school closures.

---

## Data sources

All data is openly available from UK government sources. No registration or API key is required.

| Dataset | Publisher | Years | Format | Link |
|---|---|---|---|---|
| School absence statistics | DfE (Explore Education Statistics) | 2018-19, 2021-22, 2022-23, 2023-24 | CSV | [explore-education-statistics.service.gov.uk](https://explore-education-statistics.service.gov.uk/find-statistics/pupil-absence-in-schools-in-england) |
| Get Information About Schools (GIAS) | DfE | Current | CSV | [get-information-schools.service.gov.uk/Downloads](https://get-information-schools.service.gov.uk/Downloads) |
| Index of Multiple Deprivation 2019 (File 7) | MHCLG | 2019 | CSV | [assets.publishing.service.gov.uk](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/845345/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv) |
| ONS Postcode Directory | ONS | Nov 2023 | CSV | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk/datasets/ons::ons-postcode-directory-november-2023/about) |

---

## Feature list

| Feature | Description | Source |
|---|---|---|
| `year_numeric` | Academic year encoded as 0 (2018-19), 3 (2021-22), 4 (2022-23), 5 (2023-24) | Derived |
| `imd_score` | IMD 2019 score (higher = more deprived) | IMD via postcode → LSOA |
| `imd_quintile` | Deprivation quintile 1–5 (1 = most deprived) | Derived from IMD decile |
| `percent_fsm` | % of pupils eligible for Free School Meals | GIAS |
| `log_pupils` | Log-transformed school roll size | GIAS |
| `phase_numeric` | 1=Primary, 2=Secondary, 3=All-through | GIAS |
| `is_urban` | 1 if urban location, 0 if rural | GIAS (ONS urban/rural class.) |
| `region_*` | One-hot encoded region dummies (9 English regions) | GIAS |

---

## Methodology

**1. Ingest.** `src/ingest.py` downloads GIAS, IMD, and the ONSPD postcode lookup automatically. The four DfE absence files require manual download from EES (no programmatic endpoint). The script prints step-by-step instructions for each year file.

**2. Clean.** `src/clean.py` loads all four absence year files and stacks them into a panel of ~85,000 school-year rows. It filters to state-funded mainstream schools (community, voluntary aided/controlled, academies, free schools) in Primary, Secondary, or All-through phases. Rows with suppressed absence values (coded `z`, `x`, or `c`) are dropped. GIAS is joined on URN, the ONSPD lookup maps school postcodes to LSOAs, and IMD scores are joined on LSOA code. A `panel_id` column (`{urn}_{year_key}`) uniquely identifies each school-year row.

**3. Feature engineering.** `src/features.py` encodes year as a numeric feature, Ofsted ratings as ordered numerics, log-transforms school size, derives deprivation quintiles from IMD deciles, creates a binary urban flag, and one-hot encodes region. Missing values are imputed with column medians. Outputs `features.csv`, `target.csv`, and `groups.csv` (school URN + region metadata used by the model for splitting).

**4. Modelling.** `src/model.py` trains a Random Forest Regressor (scikit-learn). Train/test split is by school URN — all years for a given school land in the same split — to prevent cross-year data leakage. GridSearchCV tunes `n_estimators`, `max_depth`, and `min_samples_split` using **spatial GroupKFold cross-validation** (9 folds, one per English region), so train and validation folds never share the same geography. An OLS regression serves as the baseline comparator.

**5. Explainability.** `src/explain.py` uses SHAP TreeExplainer to compute Shapley values on the test set. A bar summary plot shows overall feature importance by mean |SHAP|. A beeswarm plot shows direction and distribution of effects. Dependence plots for the top features show how each predictor's effect varies across its range, coloured by IMD score.

---

## Model performance

| Model | Test RMSE | Test R² | CV RMSE (spatial, 9-fold) |
|---|---|---|---|
| OLS baseline | 6.446 | 0.486 | — |
| Random Forest | 5.706 | 0.597 | 5.604 ± 0.047 |

The R² improvement from 0.547 (single-year) to 0.597 (panel) is achieved under a stricter evaluation setup — school-level splitting and spatial CV both make the test harder, not easier. The panel's additional temporal variation is the source of the gain.

---

## Key findings from SHAP

- **FSM remains the strongest predictor.** Free School Meals eligibility (`percent_fsm`, importance 0.397) is the dominant feature. Schools with higher FSM rates receive large positive SHAP contributions — individual-level poverty is a more precise signal than the area-level IMD score.

- **Year is the second most important feature.** `year_numeric` (importance 0.298) captures the post-COVID absence surge. The SHAP dependence plot shows a sharp structural break: schools in 2018-19 receive large negative contributions (–5 to –10), meaning absence was much lower than the panel average. The 2021-22 cohort — the first full post-lockdown year — receives the highest positive contributions (~+3 to +4), reflecting the attendance crisis. By 2022-23 and 2023-24 values are declining but remain positive, indicating absence has not returned to pre-pandemic levels. Crucially, the IMD colour coding shows this year effect is **uniform across deprivation levels** — the COVID attendance shock was not disproportionately concentrated in deprived schools.

- **School phase is third.** `phase_numeric` (importance 0.209) reflects that secondary schools have substantially higher persistent absence than primary schools, independent of deprivation and time.

- **Area deprivation is weaker than expected.** `imd_score` (importance 0.021) ranks fifth, well behind FSM. This confirms that targeting interventions by area-level deprivation alone misses the highest-risk pupils; FSM eligibility is the sharper targeting criterion.

- **RF vs OLS.** The Random Forest (Test R² = 0.597, RMSE = 5.706) outperforms OLS (Test R² = 0.486, RMSE = 6.446) by a larger margin than in the single-year model. The panel's temporal structure introduces non-linear interactions — particularly the year × phase interaction — that OLS cannot capture.

---

## Policy implications

The year SHAP finding is the most policy-relevant result from the panel extension. The COVID absence surge hit schools uniformly regardless of deprivation level — which means area-targeted interventions would not have been the right response. A universal, time-bound attendance recovery programme would have been more efficient than one targeted by deprivation decile.

The persistence of elevated absence in 2022-23 and 2023-24 (both years still positive SHAP) suggests the disruption was not a one-year shock but a structural shift in attendance norms. Whether that shift is reversing or stabilising at a new higher level will require the next annual DfE release to determine.

The FSM finding strengthens the case for pupil-level targeting: DfE and local authorities already have FSM eligibility as an administrative flag. Using it directly to prioritise attendance outreach is both more precise and more actionable than IMD-based area targeting.

---

## Limitations

- **IMD vintage.** IMD 2019 predates the pandemic. Post-pandemic patterns of disadvantage may not be fully captured by 2019 deprivation scores, potentially understating the deprivation effect in the 2022-24 years.
- **Suppressed data gaps.** DfE suppresses absence rates for very small schools, so the smallest rural schools are systematically excluded.
- **No 2019-20 or 2020-21 data.** The COVID years are absent from the panel, creating a gap in the time series that `year_numeric` encodes but cannot fully account for.
- **Static school characteristics.** GIAS attributes (FSM, Ofsted, size) are from the current extract, not matched year-by-year. Schools that changed phase, size, or Ofsted grade across the panel period are assigned a single value.
- **Label leakage risk.** School size and Ofsted grade could in theory be affected by high absence. The causal direction is not established by this model.

---

## How to reproduce

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/school-absence-deprivation-ml.git
cd school-absence-deprivation-ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download raw data

```bash
python src/ingest.py
```

GIAS, IMD, and ONSPD download automatically. The four DfE absence files require manual download — the script prints instructions for each year. Rename and save as:

```
data/raw/absence_1819_school.csv   # 2018-19: file named *nat_reg_la_sch*
data/raw/absence_2122_school.csv   # 2021-22: file named *1_absence_3term_school*
data/raw/absence_2223_school.csv   # 2022-23: file named *1_absence_3term_school*
data/raw/absence_2324_school.csv   # 2023-24: file named *1_absence_3term_school*
```

### 3. Clean and join

```bash
python src/clean.py
```

Outputs `data/processed/schools_clean.csv` (~79,000 school-year rows).

### 4. Feature engineering

```bash
python src/features.py
```

Outputs `data/processed/features.csv`, `data/processed/target.csv`, and `data/processed/groups.csv`.

### 5. Train and evaluate

```bash
python src/model.py
```

Outputs `outputs/model_metrics.json`, `outputs/feature_importances.csv`, and `models/random_forest.pkl`.

### 6. SHAP explainability

```bash
python src/explain.py
```

Outputs SHAP plots to `outputs/figures/`.

### 7. Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Repository structure

```
school-absence-deprivation-ml/
├── data/
│   ├── raw/              # gitignored - populated by ingest.py + manual downloads
│   └── processed/        # generated by clean.py and features.py
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── ingest.py         # data download
│   ├── clean.py          # filtering, joining, panel construction
│   ├── features.py       # feature engineering + groups metadata
│   ├── model.py          # Random Forest + OLS baseline, spatial CV
│   └── explain.py        # SHAP plots
├── outputs/
│   └── figures/          # SHAP plots and model charts
├── models/               # gitignored - saved model pickle
├── requirements.txt
├── .gitignore
└── README.md
```
