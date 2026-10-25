# Lifestyle Health Check — Full-Stack ML with CI/CD

Binary classifier (`healthy` vs `diseased`) trained on `health_lifestyle_classification.csv`
using the ensemble method from:

> Huang W., Ying T.W., Chin W.L.C. *et al.* "Application of ensemble machine learning algorithms on
> lifestyle factors and wearables for cardiovascular risk prediction." *Scientific Reports* 12, 1033 (2022).

Python **Flask** API runs the model locally; an **Angular** single-page form calls it; **GitHub Actions**
tests the code and publishes the Angular page to **GitHub Pages** on every push to `main`.

## Model

| Item | Choice (following the paper's classification phase) |
|---|---|
| Task | Binary classification, target column `target` (positive class = `diseased`, 29.9 %) |
| Base learners | Gaussian Naive Bayes, Random Forest, linear Support Vector Classifier |
| Ensemble | Majority (hard) vote |
| Class imbalance | SMOTE oversampling, applied to training folds only (`imblearn` pipeline) |
| Validation | Stratified 5-fold cross-validation (80 / 20) |
| Metrics | Sensitivity, specificity, PPV, NPV, F1, AUC (mean, SD, 95 % CI) |
| Feature importance | Absolute linear-SVC coefficients |
| Preprocessing | Median imputation + standard scaling (numbers); `"None"` imputation + one-hot (categories) |
| Inputs | 37 raw features (20 numeric, 17 categorical) |

Not reproduced: the paper's regression phase and decision node (they need a continuous calcium score,
which this dataset does not have) and the Framingham comparison (needs HDL / diabetes columns).

Dropped columns: `survey_code` (ID), `electrolyte_level`, `environmental_risk_score`,
`gene_marker_flag` (constants), `bmi_estimated`, `bmi_scaled`, `bmi_corrected`, `height`, `weight`
(duplicates of `bmi`), `daily_supplement_dosage` (negative dosages).

### Cross-validated results (5-fold)

| Model | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|
| **Ensemble** | 0.169 | 0.832 | 0.216 | **0.500** |
| Naive Bayes | 0.243 | 0.757 | 0.268 | 0.500 |
| Random Forest | 0.000 | 1.000 | 0.000 | 0.500 |
| Linear SVC | 0.495 | 0.508 | 0.374 | 0.501 |

**Important:** AUC ≈ 0.50 means the predictions are no better than chance. Every feature has a
correlation with the target below 0.011 and the disease rate is ~30 % in every category, so this dataset
has no learnable signal. The pipeline is correct; the data cannot support a useful model. Full numbers:
`docs/model_metrics.json`.

## Run it on your computer (Windows)

Needs Python 3.12+, Node.js 22+ (LTS) and Git.

**Easiest:** double-click in this order
1. `setup.bat` — creates `.venv`, installs Python and Angular packages (once)
2. `start_backend.bat` — Flask on http://127.0.0.1:5000 (leave the window open)
3. `start_frontend.bat` — Angular on http://localhost:4200 (leave the window open, opens the browser)

**Manual commands** (PowerShell, from the project folder):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python backend\test_pipeline.py          # direct inference -> PASS: local inference
python -m pytest -q backend\tests        # 7 passed
python backend\app.py                    # Terminal 1: Flask (keep open)

cd frontend; npm ci; npx ng serve --open # Terminal 2: Angular (proxy to Flask is built in)
```

Test the API: `curl.exe --noproxy "*" -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary "@backend/test_payload.json"`

### Retraining

Put the full CSV at `data/raw/health_lifestyle_classification.csv` (not committed, 53 MB), then:

```powershell
python backend\train_model.py            # ~2 min; writes backend\models\model_bundle.joblib
python backend\make_payload.py           # writes backend\test_payload.json
python backend\export_frontend_schema.py # keeps the Angular form in sync with the model
```

## API

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Status, feature count, model version |
| `/schema` | GET | Feature names, ranges and allowed categories |
| `/predict` | POST | JSON with all 37 features → `prediction`, each model's `votes`, `agreement` |

The API rejects missing or extra keys, non-numbers, out-of-range numbers and unknown categories (HTTP 400).

## CI/CD

- `.github/workflows/ci.yml` — backend pytest (Python 3.12) and Angular unit tests + build on every push / PR.
- `.github/workflows/pages.yml` — builds Angular with `--base-href /<repo-name>/` (filled in
  automatically) and deploys `frontend/dist/frontend/browser` to GitHub Pages with `actions/deploy-pages@v4`.

GitHub Pages hosts only the static Angular page. It shows the notice *"Static GitHub Pages
Demonstration - Local Flask backend required for live inference."* and disables the check button,
because Pages cannot run Python.

## Project layout

```
backend/   app.py, train_model.py, make_payload.py, export_frontend_schema.py,
           test_pipeline.py, test_api.py, test_payload.json, requirements.txt,
           models/model_bundle.joblib, tests/test_api.py, tests/test_structure.py
data/      sample_inputs.csv (1 row)          docs/   model_metrics.json
frontend/  Angular 22 app (src/app/app.ts, app.html, app.css, feature-schema.ts,
           prediction.service.ts), proxy.conf.json
.github/workflows/  ci.yml, pages.yml
```

Research and teaching prototype only. Not a medical device and not medical advice.
