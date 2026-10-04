# AI651 — Deep Learning for Space, Time and Graphs · Programming Assignment 1

LUMS, Fall 2026. The report is in `report/` (LaTeX source and figures).

```
Question 1/                     Task 1: forecasting across heterogeneous sensors
  Assignment1.ipynb             notebook with the three implementations and deployment choices
  Assignment1_executed_full.ipynb   executed with PA1_PRESET=full (submitted evidence)
  harness/                      supplied course harness (unchanged)
  results/design/               numbered Outputs 1.1–4.3 (CSV, LaTeX tables, PDF figures)
  requirements.txt
Question 2 - Leaderboard/       Task 2: Autoformer leaderboard challenge
  Data/                         provided CSVs
  autoformer.py                 Autoformer (series decomposition + Auto-Correlation)
  pipeline.py                   data, chronological validation, multi-seed training, final forecast
  eda_figure.py                 exploratory figure used in the report
  results/                      per-experiment JSON + logs, final forecast and submission string
report/                         main.tex + figures/
```

## Environment

Python 3.10+ with PyTorch 2.x (developed in a conda env with Python 3.10.19, torch 2.6.0+cu124).

```bash
pip install -r "Question 1/requirements.txt"
```

## Task 1

Implemented in `Question 1/Assignment1.ipynb`: `RawAttentionForecaster.forward`,
`SeriesDecomposition.forward`, and `aggregate_delays`. Each one passes its notebook check.

```bash
cd "Question 1"
PA1_PRESET=full jupyter nbconvert --to notebook --execute Assignment1.ipynb \
    --output Assignment1_executed_full.ipynb --ExecutePreprocessor.timeout=-1
```

Deployment choices (made from validation evidence before Output 4.3 was run): Period-routed ridge
for the established stations and for held-out Station 4.

## Task 2

The provided data is in `Question 2 - Leaderboard/Data/`: `student_train.csv`, `student_test.csv`
and `optional_external_data.csv`.

```bash
cd "Question 2 - Leaderboard"
python pipeline.py --baselines --tag baselines                 # naive baselines
python pipeline.py --exog none --stride 4 --tag ablation_none  # external-data ablation (3 seeds)
python pipeline.py --exog past --stride 4 --tag ablation_past
python pipeline.py --exog full --stride 4 --tag ablation_full
python pipeline.py --exog full --stride 4 --target log --tag tune_log   # chosen config (3 seeds)
python diagnose_stability.py && python diagnose_ensemble.py         # forecast-stability checks
# Leaderboard attempt 1: single refit model, P = 22081, E = 7 + 5 = 12
python pipeline.py --final --refit --exog full --target log --stride 4 --max_epochs 10 \
    --patience 2 --seeds 0 --tag final
# Leaderboard attempt 2 (best): 3-seed ensemble, no refit, P = 66243, E = 7 + 6 + 4 = 17
python pipeline.py --final --exog full --target log --stride 4 --max_epochs 10 \
    --patience 2 --seeds 0 1 2 --tag final_ensemble
```

Each `--final` command prints the declared **P** (trainable parameters) and **E** (training
epochs). For an ensemble, both are summed over its members, and E includes any refit epochs. The
168 comma-separated forecasts are written to `results/<tag>_submission.txt`.

### Attribution

`autoformer.py` was written from scratch following Wu et al. (2021), *Autoformer: Decomposition
Transformers with Auto-Correlation for Long-Term Series Forecasting*, and the structure of the
authors' reference implementation, https://github.com/thuml/Autoformer. The module docstring lists
the differences from that implementation.

### Leaderboard attempt 3 (best): log + raw ensemble

```bash
cd "Question 2 - Leaderboard"
python ensemble_members.py --targets log raw --seeds 0 1 2 3 4   # 10 validated members
python ensemble_eval.py                                           # ensembles vs attempt 2 (bootstrap)
python ensemble_weight.py                                         # log/raw mixing weight scan
python make_submission.py --raw_weight 0.3 --tag mix_w03          # P = 220810, E = 47
```

| Attempt | Model | Val RMSE | Leaderboard RMSE | P | E |
|---|---|---|---|---|---|
| 1 | single log model + refit | (not validated) | 142.04 | 22,081 | 12 |
| 2 | log ×3 ensemble | 63.95 | 112.15 | 66,243 | 17 |
| 3 | 0.7·log ×5 + 0.3·raw ×5 | 62.05 | **102.21** | 220,810 | 47 |
