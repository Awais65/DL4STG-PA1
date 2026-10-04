"""Build a leaderboard submission from saved ensemble members (see ensemble_members.py).

    python make_submission.py --log 0 1 2 3 4 --raw 0 1 2 3 4 --raw_weight 0.3 --tag mix_w03

forecast = (1 - raw_weight) * mean(log members) + raw_weight * mean(raw members).
P and E are summed over every member used.
"""
import argparse
import json

import numpy as np
import pandas as pd

import pipeline as P

cli = argparse.ArgumentParser()
cli.add_argument("--log", type=int, nargs="*", default=[0, 1, 2, 3, 4])
cli.add_argument("--raw", type=int, nargs="*", default=[0, 1, 2, 3, 4])
cli.add_argument("--raw_weight", type=float, default=0.3)
cli.add_argument("--tag", default="mix_w03")
opts = cli.parse_args()

M = P.RESULTS / "members"
members = {f"log_s{s}": np.load(M / f"log_s{s}.npz") for s in opts.log}
members.update({f"raw_s{s}": np.load(M / f"raw_s{s}.npz") for s in opts.raw})
part = lambda prefix, key: np.mean([m[key] for n, m in members.items() if n.startswith(prefix)], 0)
w = opts.raw_weight if opts.raw else 0.0
blend = lambda key: (1 - w) * part("log", key) + w * part("raw", key) if opts.raw else part("log", key)

forecast = blend("hidden")
assert forecast.shape == (P.PRED_LEN,) and np.isfinite(forecast).all() and (forecast >= 0).all()
val = P.metrics(members[next(iter(members))]["actual"], blend("val"))
P_total = int(sum(int(m["params"]) for m in members.values()))
E_total = int(sum(int(m["epochs"]) for m in members.values()))

pd.DataFrame({"time_idx": np.arange(P.N_TRAIN + 1, P.N_FULL + 1), "value": forecast}).to_csv(
    P.RESULTS / f"{opts.tag}_forecast.csv", index=False)
(P.RESULTS / f"{opts.tag}_submission.txt").write_text(",".join(f"{v:.4f}" for v in forecast) + "\n")
(P.RESULTS / f"{opts.tag}.json").write_text(json.dumps(
    {"members": list(members), "raw_weight": w, "P": P_total, "E": E_total, "val": val,
     "epochs_per_member": {n: int(m["epochs"]) for n, m in members.items()}}, indent=1))
print(f"val RMSE {val['RMSE']:.2f} MAE {val['MAE']:.2f} sMAPE {val['sMAPE']:.2f}")
print(f"P = {P_total}\nE = {E_total}\nforecast mean {forecast.mean():.1f}, median "
      f"{np.median(forecast):.1f}, max {forecast.max():.1f}")
print(f"written results/{opts.tag}_submission.txt")
