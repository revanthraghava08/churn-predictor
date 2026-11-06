# Customer Churn Predictor

A machine-learning project that predicts which telecom customers are likely to cancel,
with a Streamlit app where the decision threshold can be tuned to business costs.

**Live app:** <https://churnvision-ai.streamlit.app/>

## Problem

Losing a customer costs more than offering a retention deal to someone who would have stayed.
The goal is a model that ranks customers by churn risk, plus a threshold that reflects that
cost trade-off instead of the default 0.5.

## Data

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle):
7,043 customers, 21 columns, about 26.5% churned. The data is not included in this repo;
download the CSV into `data/` to retrain.

## Approach

1. Cleaning: `TotalCharges` is stored as text with 11 blanks, converted to numeric and median-imputed.
2. Preprocessing in a scikit-learn `Pipeline`: scaling for numeric columns, one-hot encoding for
   categorical columns (so nothing leaks from validation folds).
3. Compared three models with 5-fold stratified cross-validation.
4. Tested two engineered features (`n_addons`, `tenure_group`).
5. Tuned the regularisation strength and chose a decision threshold from out-of-fold probabilities.
6. Evaluated once on a held-out test set (20%, stratified).

## Results

### Model comparison (5-fold CV on all rows)

| Model | AUC (mean ± std) | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic regression | 0.847 ± 0.012 | 0.530 | 0.666 | 0.590 |
| Hist gradient boosting | 0.834 ± 0.009 | 0.525 | 0.639 | 0.576 |
| Random forest | 0.823 ± 0.012 | 0.492 | 0.631 | 0.553 |

Recall, precision and F1 are at the default 0.5 threshold. Logistic regression and gradient
boosting are within about one standard deviation on AUC, so the simpler model was chosen.

### Feature engineering and tuning

On the training split, neither engineered feature improved AUC by more than the fold-to-fold
noise (about 0.011), so both were left out. Logistic regression's `C` was flat from 0.1 to 10
(AUC about 0.848); `C = 0.1` was used (strongest regularisation on the plateau).

### Threshold

The cost-minimising threshold follows from the costs: flag a customer when
`p > cost_false_alarm / (cost_false_alarm + cost_missed_churner)`.
With the placeholder costs below, that is about 0.09, and out-of-fold results agreed (best at 0.10).

### Final test set (1,409 customers, threshold 0.10)

| Metric | Value |
|---|---|
| AUC | 0.841 |
| Recall | 0.952 (356 of 374 churners caught) |
| Precision | 0.400 |
| Customers flagged | 889 (63%) |

Confusion matrix `[[502, 533], [18, 356]]`.

Under *assumed* costs of 500 per missed churner and 50 per wasted offer:

| Strategy | Total cost on test set |
|---|---|
| Flag nobody | 187,000 |
| Flag everybody | 51,750 |
| This model at 0.10 | 35,650 |

The model beats doing nothing by a wide margin and beats blanket offers by about 31%.

## Limitations

- The costs (500 and 50) are placeholders, not measured business figures. The best threshold
  moves with them, which is why the app exposes a slider.
- The cost model assumes every flagged churner can be saved, which overstates the benefit.
- The app estimates `TotalCharges` as tenure × monthly charges.
- One public sample dataset, no time dimension: not validated for real deployment.

## Run locally

```bash
pip install -r requirements.txt
# put the CSV at data/WA_Fn-UseC_-Telco-Customer-Churn.csv, then:
python train.py          # trains, prints the test result, saves model.joblib
streamlit run app.py
```

## Files

- `train.py`: feature code, pipeline, final evaluation, saves the model
- `app.py`: Streamlit app (uses the same `add_features` as training)
- `model.joblib`, `model_meta.json`: trained pipeline and its settings
- `requirements.txt`: pinned versions (needed to load the saved model)