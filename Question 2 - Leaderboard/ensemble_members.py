"""Train validated (early-stopped, no refit) ensemble members and save their predictions.

For each target transform and seed, saves validation-block forecasts, the hidden-origin forecast,
epochs run and parameter count to results/members/<target>_s<seed>.npz. Ensembles are then scored
offline by ensemble_eval.py without retraining.

    python ensemble_members.py --targets log raw --seeds 0 1 2 3 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import pipeline as P

cli = argparse.ArgumentParser()
cli.add_argument("--targets", nargs="+", default=["log", "raw"])
cli.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
opts = cli.parse_args()

out_dir = P.RESULTS / "members"
out_dir.mkdir(parents=True, exist_ok=True)
y, external = P.load_series()
for target in opts.targets:
    sys.argv = ["x", "--exog", "full", "--target", target, "--stride", "4", "--max_epochs", "10",
                "--patience", "2"]
    args = P.parse()
    val_start, train_origins, val_origins = P.split_origins(args)
    actual = y[val_origins[:, None] + np.arange(P.PRED_LEN)]
    scaler = P.TargetScaler(y[:val_start], target)
    marks, n_exog = P.build_marks(external, val_start, args)
    windows = P.Windows(scaler.scale(y), marks, n_exog, args)
    for seed in opts.seeds:
        path = out_dir / f"{target}_s{seed}.npz"
        if path.exists():
            print(f"skip {path.name} (exists)")
            continue
        P.set_seed(seed)
        model = P.make_model(marks.shape[1], args)
        ran, best, _ = P.train_epochs(model, windows, train_origins, args, seed, args.max_epochs,
                                      (val_origins, actual, scaler))
        val = P.predict(model, windows, val_origins, scaler)
        hidden = P.predict(model, windows, np.array([P.N_TRAIN]), scaler)[0]
        np.savez(path, val=val, hidden=hidden, actual=actual, origins=val_origins,
                 epochs=ran, best_epoch=best, params=P.count_params(model))
        m = P.metrics(actual, val)
        print(f"[{target} s{seed}] RMSE {m['RMSE']:.2f} MAE {m['MAE']:.2f} epochs {ran} "
              f"hidden mean {hidden.mean():.1f}", flush=True)
