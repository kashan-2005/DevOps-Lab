"""Train the paper-style ensemble classifier and export backend/models/model_bundle.joblib.

Methodology follows Huang et al. (2022), Scientific Reports 12:1033
"Application of ensemble machine learning algorithms on lifestyle factors and
wearables for cardiovascular risk prediction" — classification phase:
  * three base learners: Naive Bayes, Random Forest, Support Vector Classifier
  * ensemble by majority (hard) vote
  * SMOTE oversampling of the minority class, applied to the TRAINING folds only
  * stratified 5-fold cross-validation (80% train / 20% hold-out per fold)
  * metrics: sensitivity, specificity, PPV, NPV, F1, AUC (mean, SD, 95% CI)
  * post-hoc feature importance from the absolute linear-SVC coefficients

Run from the project root:   python backend/train_model.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "health_lifestyle_classification.csv"
OUT = ROOT / "backend" / "models" / "model_bundle.joblib"
METRICS_OUT = ROOT / "docs" / "model_metrics.json"
SEED = 42

TARGET = "target"
POSITIVE = "diseased"

# Columns removed before modelling (see docs/model_metrics.json -> "dropped_columns")
DROPPED = {
    "survey_code": "row identifier",
    "electrolyte_level": "constant (always 0)",
    "environmental_risk_score": "constant (always 5.5)",
    "gene_marker_flag": "constant (always 1 or missing)",
    "bmi_estimated": "exact duplicate of bmi",
    "bmi_scaled": "bmi multiplied by 3",
    "bmi_corrected": "noisy duplicate of bmi",
    "height": "already captured by bmi",
    "weight": "already captured by bmi",
    "daily_supplement_dosage": "invalid values (negative dosages)",
}

NUMERIC = [
    "age", "bmi", "waist_size", "blood_pressure", "heart_rate", "cholesterol",
    "glucose", "insulin", "sleep_hours", "work_hours", "physical_activity",
    "daily_steps", "calorie_intake", "sugar_intake", "water_intake", "screen_time",
    "stress_level", "mental_health_score", "income", "meals_per_day",
]
CATEGORICAL = [
    "gender", "sleep_quality", "alcohol_consumption", "smoking_level",
    "mental_health_support", "education_level", "job_type", "occupation",
    "diet_type", "exercise_type", "device_usage", "healthcare_access",
    "insurance", "sunlight_exposure", "caffeine_intake", "family_history",
    "pet_owner",
]
FEATURES = NUMERIC + CATEGORICAL
MISSING_CATEGORY = "None"  # blank alcohol / caffeine / exercise answers mean "none"


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", SkPipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", SkPipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    ensemble = VotingClassifier(
        estimators=[
            ("naive_bayes", GaussianNB()),
            ("random_forest", RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=5,
                n_jobs=-1, random_state=SEED)),
            ("svc", LinearSVC(C=1.0, dual=False, random_state=SEED)),
        ],
        voting="hard",  # majority vote, as in the paper
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("smote", SMOTE(random_state=SEED)),  # applied during fit only
        ("ensemble", ensemble),
    ])


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ppv": tp / (tp + fp) if tp + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # Hard-vote labels -> single-threshold ROC, the same way the paper's curves are drawn
        "auc": roc_auc_score(y_true, y_pred),
    }


def summarise(per_fold: list[dict]) -> dict:
    out = {}
    for key in per_fold[0]:
        vals = np.array([m[key] for m in per_fold])
        mean, sd = vals.mean(), vals.std(ddof=1)
        half = 1.96 * sd / np.sqrt(len(vals))
        out[key] = {"mean": round(mean, 4), "sd": round(sd, 4),
                    "ci95": [round(mean - half, 4), round(mean + half, 4)]}
    return out


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(DATA)
    X = df[FEATURES].copy()
    y_text = df[TARGET].astype(str)
    y = (y_text == POSITIVE).astype(int).to_numpy()
    print(f"Loaded {len(df):,} rows, {len(FEATURES)} features; "
          f"{POSITIVE} = {y.mean():.1%}")

    # ---- stratified 5-fold cross-validation (SMOTE inside each training fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    per_fold = {"ensemble": [], "naive_bayes": [], "random_forest": [], "svc": []}
    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        pipe = build_pipeline().fit(X.iloc[tr], y[tr])
        per_fold["ensemble"].append(fold_metrics(y[te], pipe.predict(X.iloc[te])))
        Xt = pipe.named_steps["preprocess"].transform(X.iloc[te])
        for name, est in pipe.named_steps["ensemble"].named_estimators_.items():
            per_fold[name].append(fold_metrics(y[te], est.predict(Xt)))
        e = per_fold["ensemble"][-1]
        print(f"  fold {k}: AUC={e['auc']:.3f} sens={e['sensitivity']:.3f} "
              f"spec={e['specificity']:.3f}  ({time.time() - t0:.0f}s)")

    cv = {name: summarise(m) for name, m in per_fold.items()}

    # ---- final model on all rows
    final = build_pipeline().fit(X, y)

    # ---- post-hoc feature importance: |linear SVC coefficients| (as in the paper)
    names = final.named_steps["preprocess"].get_feature_names_out()
    coefs = np.abs(final.named_steps["ensemble"].named_estimators_["svc"].coef_[0])
    top = sorted(zip(names, coefs), key=lambda t: -t[1])[:15]
    importance = [{"feature": n.split("__", 1)[1], "importance": round(float(c), 4)}
                  for n, c in top]

    # ---- input schema for the API and the Angular form
    schema = {}
    for col in NUMERIC:
        s = df[col]
        schema[col] = {"kind": "number",
                       "min": round(float(s.min()), 2), "max": round(float(s.max()), 2),
                       "median": round(float(s.median()), 2),
                       "integer": bool((s.dropna() % 1 == 0).all())}
    for col in CATEGORICAL:
        opts = sorted(df[col].dropna().unique().tolist())
        if df[col].isna().any():
            opts.append(MISSING_CATEGORY)
        schema[col] = {"kind": "category", "options": opts}

    bundle = {
        "pipeline": final,
        "features": FEATURES,
        "schema": schema,
        "task": "classification",
        "classes": {"0": "healthy", "1": POSITIVE},
        "positive_label": POSITIVE,
        "model_version": "1.0",
        "methodology": "Huang et al. 2022 (Sci Rep 12:1033): NB + RF + SVC "
                       "majority-vote ensemble, SMOTE, stratified 5-fold CV",
        "sklearn_version": sklearn.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT, compress=3)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps({
        "rows": len(df), "positive_rate": round(float(y.mean()), 4),
        "features": FEATURES, "dropped_columns": DROPPED,
        "cross_validation_5fold": cv, "top15_svc_importance": importance,
    }, indent=2), encoding="utf-8")

    print("\nCross-validated ensemble performance (positive = diseased):")
    for k, v in cv["ensemble"].items():
        print(f"  {k:12s} {v['mean']:.3f}  (95% CI {v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}, SD {v['sd']:.3f})")
    print(f"\nSaved {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
