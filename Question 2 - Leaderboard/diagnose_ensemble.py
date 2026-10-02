"""Validation check: does averaging the 3 seed models beat a single model, pooled and per block?"""
import sys
import numpy as np
import pipeline as P

sys.argv = ["x", "--exog", "full", "--target", "log", "--stride", "4", "--max_epochs", "10",
            "--patience", "2"]
args = P.parse()
y, external = P.load_series()
val_start, tr, va = P.split_origins(args)
actual = y[va[:, None] + np.arange(P.PRED_LEN)]
scaler = P.TargetScaler(y[:val_start], "log")
marks, n_exog = P.build_marks(external, val_start, args)
win = P.Windows(scaler.scale(y), marks, n_exog, args)
preds, hidden = [], []
for seed in [0, 1, 2]:
    P.set_seed(seed)
    model = P.make_model(marks.shape[1], args)
    P.train_epochs(model, win, tr, args, seed, 10, (va, actual, scaler))
    preds.append(P.predict(model, win, va, scaler))
    hidden.append(P.predict(model, win, np.array([P.N_TRAIN]), scaler)[0])
def block(f): return np.sqrt(((f - actual) ** 2).mean(1))
print()
for i, f in enumerate(preds):
    m = P.metrics(actual, f); b = block(f)
    print(f"seed{i}: RMSE {m['RMSE']:.2f} MAE {m['MAE']:.2f} sMAPE {m['sMAPE']:.1f} | block p50/p90/p97 "
          f"{np.median(b):.0f}/{np.quantile(b,.9):.0f}/{np.quantile(b,.97):.0f}")
ens = np.mean(preds, 0); m = P.metrics(actual, ens); b = block(ens)
print(f"ENSEMBLE: RMSE {m['RMSE']:.2f} MAE {m['MAE']:.2f} sMAPE {m['sMAPE']:.1f} | block p50/p90/p97 "
      f"{np.median(b):.0f}/{np.quantile(b,.9):.0f}/{np.quantile(b,.97):.0f}")
wins = np.mean([block(ens) < block(f) for f in preds])
print(f"ensemble block RMSE lower than a single seed's in {wins:.0%} of (block, seed) pairs")
h = np.mean(hidden, 0)
print(f"ensemble hidden-origin forecast: mean {h.mean():.1f} median {np.median(h):.1f} max {h.max():.1f}")
np.save("results/ensemble_hidden.npy", h)
