# Contributing

Contributions are welcome. Typical improvements include:
- additional baselines (Logistic Regression, Linear SVM, etc.)
- better preprocessing / ablation studies
- evaluation improvements (stratified CV, calibration, error analysis)
- packaging and CI

## Development setup
1. Fork the repository and create a feature branch.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Keep code formatted and add concise documentation for any new modules.

## Pull request checklist
- Clear description of what changed and why
- Reproducible commands to run training/evaluation
- No dataset files committed (keep them under `data/raw/`)
