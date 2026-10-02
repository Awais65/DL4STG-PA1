"""How stable is the hidden-origin forecast across seeds, with and without refitting on the full
history? Uses no leaderboard information."""
import sys
import numpy as np
import pipeline as P

sys.argv = ["x", "--exog", "full", "--target", "log", "--stride", "4", "--max_epochs", "10",
            "--patience", "2"]
args = P.parse()
y, external = P.load_series()
val_start, tr, va = P.split_origins(args)
actual = y[va[:, None] + np.arange(P.PRED_LEN)]
rows = {}
for seed in [0, 1, 2]:
    P.set_seed(seed)
    scaler = P.TargetScaler(y[:val_start], "log")
    marks, n_exog = P.build_marks(external, val_start, args)
    win = P.Windows(scaler.scale(y), marks, n_exog, args)
    model = P.make_model(marks.shape[1], args)
    ran, best, _ = P.train_epochs(model, win, tr, args, seed, 10, (va, actual, scaler))
    rows[f"seed{seed} validated (E={ran})"] = P.predict(model, win, np.array([P.N_TRAIN]), scaler)[0]
    P.set_seed(seed)
    scaler = P.TargetScaler(y[:P.N_TRAIN], "log")
    marks, n_exog = P.build_marks(external, P.N_TRAIN, args)
    win = P.Windows(scaler.scale(y), marks, n_exog, args)
    model = P.make_model(marks.shape[1], args)
    P.train_epochs(model, win, np.arange(args.seq_len, P.N_TRAIN - P.PRED_LEN + 1, 4), args, seed, best)
    rows[f"seed{seed} refit {best} ep (E={ran + best})"] = P.predict(model, win, np.array([P.N_TRAIN]), scaler)[0]
print()
for k, f in rows.items():
    print(f"{k:28s} mean {f.mean():6.1f}  median {np.median(f):6.1f}  max {f.max():6.1f}")
F = np.array(list(rows.values()))
np.save("results/stability_forecasts.npy", F)
print("pairwise RMSE between these forecasts: mean", round(float(np.mean(
    [np.sqrt(((a - b) ** 2).mean()) for i, a in enumerate(F) for b in F[i + 1:]])), 1))
