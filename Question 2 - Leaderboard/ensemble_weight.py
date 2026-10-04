"""Validation scan of the log/raw mixing weight w (forecast = (1-w)*log_mean + w*raw_mean).

Uses saved members only. The 90% interval for the best w comes from a week-block bootstrap: for
each resample, w is re-chosen, which shows how stable the choice is.
"""
import numpy as np

import pipeline as P

M = P.RESULTS / "members"
log = np.mean([np.load(M / f"log_s{i}.npz")["val"] for i in range(5)], 0)
raw = np.mean([np.load(M / f"raw_s{i}.npz")["val"] for i in range(5)], 0)
actual = np.load(M / "log_s0.npz")["actual"]
ws = np.round(np.arange(0, 1.01, 0.1), 1)
sq = {w: (((1 - w) * log + w * raw - actual) ** 2).mean(1) for w in ws}
for w in ws:
    m = P.metrics(actual, (1 - w) * log + w * raw)
    print(f"w={w:.1f}  RMSE {m['RMSE']:6.2f}  MAE {m['MAE']:5.1f}  sMAPE {m['sMAPE']:5.1f}")
groups = np.arange(len(actual)) // 7
rng = np.random.default_rng(0)
best = []
for _ in range(2000):
    idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.integers(0, groups.max() + 1, groups.max() + 1)])
    best.append(min(ws, key=lambda w: sq[w][idx].mean()))
best = np.array(best)
print("bootstrap choice of w:", {float(w): round(float((best == w).mean()), 3) for w in ws if (best == w).any()})
