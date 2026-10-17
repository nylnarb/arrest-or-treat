# Arrest or Treat? Measuring How US States Respond to Drug Use

**Which US states arrest people for drug use, and which states treat them? And what explains the difference?**

I built a **Criminalization Index** for 49 states over 2015–2022 by joining 7 federal data sources. Then I trained ML models to explain why some states lean much harder on arrests than others. Idaho's average index is about **50× higher** than Massachusetts' or Connecticut's.

![SHAP beeswarm](outputs/shap_beeswarm.png)

---

## Key Findings

- **Idaho is the outlier.** Its average index is 6.66, meaning it made about 6.7 drug arrests for every treatment admission. Massachusetts, Connecticut and Vermont are near 0.13, or about 1 arrest per 8 admissions.
- **Politics and incarceration predict it best.** The strongest single predictor is the interaction between Republican presidential vote share and incarceration rate. Region comes next: Northeast and Midwest states score lower.
- **Legal recreational marijuana goes with less criminalization.** In the OLS model the coefficient is −0.42 (p < 0.01).
- **Overdose deaths and poverty don't push states toward arrests.** States with more of either score lower once the other factors are controlled for.
- **Drug arrests fell 43% nationally from 2015 to 2022** (402 → 228 per 100k). Treatment admissions fell less, by 33%.

## Results

5-fold cross-validated performance on 377 clean state-year observations (target: log of the index):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| OLS Linear Regression | 0.41 | 0.580 | 0.839 |
| Random Forest | 0.62 | 0.443 | 0.678 |
| Extra Trees | 0.64 | 0.420 | 0.655 |
| XGBoost | 0.65 | 0.391 | 0.649 |
| Gradient Boosting | 0.66 | 0.385 | 0.636 |
| **XGBoost (tuned, 60-iter random search)** | **0.68** | — | — |

The tree models beat linear regression by about 25 R² points, which suggests strong non-linear and interaction effects. I kept OLS as the model to interpret, because it has coefficients and p-values, and used SHAP to explain the tree models.

![Feature importance and predicted vs. actual](outputs/fig3_feature_importance.png)

---

## The Metric

```
Criminalization Index = drug arrests per 100k ÷ drug treatment admissions per 100k
```

- **> 1.0**: more drug arrests than treatment admissions (criminalization-oriented)
- **< 1.0**: more treatment admissions than arrests (treatment-oriented)
- The national median is **0.996**, almost exactly balanced. The distribution is heavily right-skewed (max 14.1, Idaho 2019), so I log-transformed the index before modeling.

## Data Sources

| Source | What it provides |
|---|---|
| FBI NIBRS / Crime Data Explorer API | Drug arrests by state-year |
| SAMHSA TEDS-A (about 14M records) | Treatment admissions, criminal-justice referral %, MAT adoption, case mix |
| SAMHSA N-SSATS / N-SUMHSS | Treatment facilities and beds per capita, ownership mix, Medicaid acceptance |
| US Census ACS API | Population, poverty, income, unemployment, race/ethnicity |
| CDC VSRR | Drug overdose death rates |
| Vera Institute / BJS | Incarceration rates |
| Election & legislature data | Presidential vote share, governor party, legislative control |

The final panel has **392 state-year rows × 41 columns**, saved in `data/processed/panel_dataset.csv`.

## Data Quality Work

Federal crime data has known reporting gaps. I identified them and excluded the affected years so they wouldn't distort the results:

| Excluded | Why |
|---|---|
| Florida 2017–2021 | Florida stopped submitting to NIBRS, so its arrest counts are artificially low |
| Oregon (all years) | Oregon decriminalized drugs (Measure 110) and its treatment reporting is incomplete |
| Idaho 2022, Illinois 2020–2021 | Artifacts from the NIBRS transition |

## Pipeline

```
Federal APIs / raw files
        ↓
  src/fetch_*.py, src/parse_*.py   each source pulled and cleaned on its own
        ↓
  src/02_data_processing.py        merge, exclusions, compute the index
        ↓
  data/processed/panel_dataset.csv
        ↓
  src/03_eda.py → src/04_modeling.py → src/05_visualizations.py
        ↓
  outputs/                         interactive Plotly maps, SHAP plots, metrics
```

`notebooks/DrugPolicyCriminalization.ipynb` runs the whole analysis interactively. Its toggles cover outlier filtering, feature sets, log transform and tuning.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels · SHAP · Plotly · matplotlib/seaborn · REST APIs (Census, FBI CDE, CDC Socrata)

## Run It

```bash
git clone https://github.com/nylnarb/arrest-or-treat.git
cd arrest-or-treat
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn statsmodels xgboost plotly seaborn matplotlib us shap

python src/03_eda.py
python src/04_modeling.py
python src/05_visualizations.py
```

The processed dataset is included, so the analysis runs without the raw files. Raw source data isn't included because of its size. See `Criminalization Vs. Treatment.md` for where to get each source.

## Limitations

- **The results are associations, not causal effects.** The models show which state traits go with a high index. They don't show that those traits cause it.
- The data is a panel of only ~49 states, so state-years from the same state are correlated. Adding state fixed effects or grouped cross-validation is the next step.
- A low arrest count can mean a state reports poorly to NIBRS rather than enforcing lightly. I handled the known gaps (listed above), but some may remain.

## Outputs

The interactive HTML maps and charts are in `outputs/`. Download one and open it in a browser, since GitHub won't render them inline.

- `viz1_choropleth_final.html`: the index mapped by state
- `viz2_time_trends_final.html`: national arrest vs. treatment trends
- `viz5_state_rankings.html`: state rankings
- `viz_residuals_choropleth.html`: states that are more or less criminalizing than the model predicts

---

**Author:** Branlyn Roberson · [LinkedIn](https://www.linkedin.com/in/branlynroberson/)
