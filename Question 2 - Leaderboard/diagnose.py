"""Diagnostic: per-block validation errors of the log- and raw-target models, and their
forecasts at the hidden origin. Uses only training/validation data."""
import sys
import numpy as np
import pipeline as P

base = ["--exog", "full", "--stride", "4", "--max_epochs", "10", "--patience", "2", "--seeds", "0"]
y, external = P.load_series()
out = {}
for target in ["log", "raw"]:
    sys.argv = ["x", *base, "--target", target]
    args = P.parse()
    val_start, tr, va = P.split_origins(args)
    scaler = P.TargetScaler(y[:val_start], args.target)
    marks, n_exog = P.build_marks(external, val_start, args)
    win = P.Windows(scaler.scale(y), marks, n_exog, args)
    actual = y[va[:, None] + np.arange(P.PRED_LEN)]
    P.set_seed(0)
    model = P.make_model(marks.shape[1], args)
    P.train_epochs(model, win, tr, args, 0, args.max_epochs, (va, actual, scaler))
    f = P.predict(model, win, va, scaler)
    hidden = P.predict(model, win, np.array([P.N_TRAIN]), scaler)[0]
    out[target] = (f, hidden)

lv = actual.mean(1)
print(f"\nvalidation blocks: {len(va)}; actual block mean ranges {lv.min():.0f}..{lv.max():.0f}")
for t, (f, hidden) in out.items():
    rm = np.sqrt(((f - actual) ** 2).mean(1))
    bias = (f - actual).mean(1)
    hi = lv > np.quantile(lv, 0.75)
    print(f"[{t}] pooled RMSE {np.sqrt(((f-actual)**2).mean()):.1f} | block RMSE p50/p90/max "
          f"{np.median(rm):.0f}/{np.quantile(rm,.9):.0f}/{rm.max():.0f} | blocks >=142: {(rm>=142).mean():.1%}")
    print(f"      mean bias all {bias.mean():+.1f} | in top-quartile-level blocks {bias[hi].mean():+.1f} "
          f"(RMSE there {np.sqrt(((f[hi]-actual[hi])**2).mean()):.1f})")
    last = slice(-60, None)  # last ~60 origins = last ~2 months (closest season to hidden block)
    print(f"      last-2-months blocks: RMSE {np.sqrt(((f[last]-actual[last])**2).mean()):.1f}, "
          f"bias {(f[last]-actual[last]).mean():+.1f}, actual mean {actual[last].mean():.0f}")
    print(f"      hidden-origin forecast mean {hidden.mean():.1f}, median {np.median(hidden):.1f}, max {hidden.max():.1f}")
