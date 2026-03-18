# Google Colab demo

This folder contains a Colab-friendly workflow for `EnergyCommunityEvaluation`.

## What is included

- `colab/demo_setup.py` – creates a tiny synthetic energy-community dataset, a matching config file, and helper functions to run the existing project scripts.
- `colab/EnergyCommunityEvaluation_Demo.ipynb` – a notebook that:
  1. installs the project in Colab,
  2. generates the demo dataset,
  3. runs `preprocessing.py`,
  4. runs a preprocessing sanity check,
  5. runs `run_parametric_evaluation.py` with a one-click command,
  6. opens key outputs for review.

## Intended audience

The notebook is designed for users who need a runnable end-to-end example without first collecting a full municipal dataset.

## Local smoke test

From the repository root you can create the demo workspace and run both main scripts with:

```powershell
python -m colab.demo_setup --workspace-root colab_runtime --run-preprocessing --run-parametric-evaluation
```

## Typical Colab flow

1. Open `colab/EnergyCommunityEvaluation_Demo.ipynb` in Google Colab.
2. Run the setup cell to clone the repository and install `requirements.txt`.
3. Run the dataset-generation cell.
4. Run preprocessing.
5. Run the sanity-check/output-preview cell.
6. Run one-click evaluation and explore generated CSV/NetCDF outputs.

## Notes

- The demo uses a synthetic municipality called `ColabTown`.
- The configuration uses the existing `preprocessing.py` and `run_parametric_evaluation.py` entry points rather than re-implementing the workflow.
- The generated REC structure JSON is included as a reference artifact.

## Troubleshooting quick checks

- If evaluation fails early, rerun preprocessing and confirm required CSV files exist in `output/ColabTown/`.
- If imports fail after a runtime reset, rerun the `pip install -r requirements.txt` cell.
- If opening CSV manually in pandas, use `sep=';'`.
