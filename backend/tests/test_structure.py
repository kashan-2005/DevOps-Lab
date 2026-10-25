from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    "README.md",
    ".gitignore",
    "backend/app.py",
    "backend/train_model.py",
    "backend/requirements.txt",
    "backend/test_payload.json",
    "backend/models/model_bundle.joblib",
    "data/sample_inputs.csv",
    "frontend/package.json",
    "frontend/proxy.conf.json",
    ".github/workflows/pages.yml",
]


def test_required_files_exist():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    assert not missing, f"Missing repository files: {missing}"
