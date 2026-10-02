"""Task 2 pipeline: data, chronological validation, multi-seed training, final forecast.

Examples
    python pipeline.py --baselines
    python pipeline.py --exog none --tag noexog
    python pipeline.py --exog full --tag full
    python pipeline.py --exog full --tag final --final --refit
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from autoformer import Autoformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
RESULTS = ROOT / "results"
PRED_LEN = 168
N_TRAIN = 43656                      # observed history
N_FULL = 43824                       # history + 168 hidden steps
CONTINUOUS = ["feature_A", "feature_B", "feature_C", "feature_D", "feature_E", "feature_F"]
SKEWED = ["feature_D", "feature_E", "feature_F"]   # cumulative, non-negative, heavy-tailed
BINARY = ["feature_G", "feature_H", "feature_I", "feature_J"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data
def load_series():
    train = pd.read_csv(DATA / "student_train.csv")
    external = pd.read_csv(DATA / "optional_external_data.csv")
    assert len(train) == N_TRAIN and len(external) == N_FULL
    assert (train.time_idx.values == np.arange(1, N_TRAIN + 1)).all()
    y = np.full(N_FULL, np.nan)
    y[:N_TRAIN] = train.value.values
    return y, external


def build_marks(external, fit_end, args):
    """Per-step covariates [N_FULL, m]. Statistics use only steps < fit_end."""
    columns = []
    if args.exog != "none":
        cont = external[args.features_cont].astype(float).copy()
        for c in SKEWED:
            if c in cont:
                cont[c] = np.log1p(cont[c].clip(lower=0))
        mu, sd = cont.iloc[:fit_end].mean(), cont.iloc[:fit_end].std() + 1e-8
        columns.append(((cont - mu) / sd).values)
        if args.features_bin:
            columns.append(external[args.features_bin].values.astype(float))
    n_exog = sum(c.shape[1] for c in columns)
    if args.daily_phase:
        # Period 24 is recovered from the spectrum/ACF of the training target; the phase is
        # simply time_idx mod 24 (no calendar information is used).
        phase = 2 * math.pi * (np.arange(N_FULL) % 24) / 24
        columns.append(np.stack([np.sin(phase), np.cos(phase)], 1))
    marks = np.concatenate(columns, 1) if columns else np.zeros((N_FULL, 0))
    return marks.astype(np.float32), n_exog


class TargetScaler:
    def __init__(self, y_fit, transform):
        self.transform = transform
        z = self._forward(y_fit)
        self.mu, self.sd = float(z.mean()), float(z.std())

    def _forward(self, y):
        return np.log1p(np.clip(y, 0, None)) if self.transform == "log" else y

    def scale(self, y):
        return (self._forward(y) - self.mu) / self.sd

    def unscale(self, z):
        z = z * self.sd + self.mu
        return np.expm1(z) if self.transform == "log" else z


class Windows:
    """Builds batches of (context, marks_enc, marks_dec, target) for given forecast origins.

    An origin o is the 0-based index of the first forecast step: context = [o-seq, o),
    target = [o, o+pred), decoder marks = [o-label, o+pred).
    """

    def __init__(self, z, marks, n_exog, args):
        self.z = torch.tensor(np.nan_to_num(z), dtype=torch.float32, device=DEVICE)
        self.marks = torch.tensor(marks, device=DEVICE)
        self.n_exog, self.args = n_exog, args
        self.ctx = torch.arange(-args.seq_len, 0, device=DEVICE)
        self.dec = torch.arange(-args.label_len, PRED_LEN, device=DEVICE)
        self.tgt = torch.arange(0, PRED_LEN, device=DEVICE)

    def batch(self, origins):
        o = torch.as_tensor(origins, device=DEVICE).view(-1, 1)
        x = self.z[o + self.ctx].unsqueeze(-1)
        y = self.z[o + self.tgt]
        marks_enc = self.marks[o + self.ctx]
        marks_dec = self.marks[o + self.dec].clone()
        if self.args.exog == "past":         # external values over the horizon are hidden
            marks_dec[:, self.args.label_len:, :self.n_exog] = 0.0
        return x, marks_enc, marks_dec, y


# ----------------------------------------------------------------------------- metrics
def metrics(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    err = forecast - actual
    denom = np.abs(actual) + np.abs(forecast)
    smape = np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0)
    block_rmse = np.sqrt((err ** 2).mean(-1)) if err.ndim == 2 else None
    out = {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
           "sMAPE": float(100 * smape.mean())}
    if block_rmse is not None:
        out["block_RMSE_mean"] = float(block_rmse.mean())
        out["block_RMSE_median"] = float(np.median(block_rmse))
    return out


def split_origins(args):
    val_start = N_TRAIN - args.val_len
    train_origins = np.arange(args.seq_len, val_start - PRED_LEN + 1, args.stride)
    val_origins = np.arange(val_start, N_TRAIN - PRED_LEN + 1, args.val_stride)
    return val_start, train_origins, val_origins


# ----------------------------------------------------------------------------- model
def make_model(n_marks, args):
    return Autoformer(args.seq_len, args.label_len, PRED_LEN, n_marks, d=args.d,
                      heads=args.heads, d_ff=args.d_ff, e_layers=args.e_layers,
                      d_layers=args.d_layers, kernel=args.kernel, factor=args.factor,
                      dropout=args.dropout).to(DEVICE)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, windows, origins, scaler, batch=512):
    model.eval()
    out = []
    for i in range(0, len(origins), batch):
        x, me, md, _ = windows.batch(origins[i:i + batch])
        out.append(model(x, me, md).squeeze(-1).cpu().numpy())
    return np.clip(scaler.unscale(np.concatenate(out)), 0, None)


def train_epochs(model, windows, origins, args, seed, epochs, val=None):
    """Train for up to `epochs`; with `val=(origins, actual, scaler)` early-stop on val RMSE.
    Returns (epochs_run, best_epoch, history)."""
    gen = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = epochs * math.ceil(len(origins) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1) if args.onecycle else None
    loss_fn = nn.MSELoss() if args.loss == "mse" else nn.HuberLoss(delta=args.huber_delta)
    best, best_state, best_epoch, history, bad = math.inf, None, 0, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = gen.permutation(origins)
        total, t0 = 0.0, time.time()
        for i in range(0, len(order), args.batch):
            x, me, md, y = windows.batch(order[i:i + args.batch])
            loss = loss_fn(model(x, me, md).squeeze(-1), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            total += loss.item() * len(x)
        record = {"epoch": epoch, "train_loss": total / len(order), "sec": time.time() - t0}
        if val is not None:
            v_origins, v_actual, scaler = val
            record.update(metrics(v_actual, predict(model, windows, v_origins, scaler)))
            if record["RMSE"] < best - 1e-6:
                best, best_epoch, bad = record["RMSE"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        history.append(record)
        print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()},
              flush=True)
        if val is not None and bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return len(history), best_epoch, history


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- experiments
def run_validation(args):
    y, external = load_series()
    val_start, train_origins, val_origins = split_origins(args)
    scaler = TargetScaler(y[:val_start], args.target)
    marks, n_exog = build_marks(external, val_start, args)
    windows = Windows(scaler.scale(y), marks, n_exog, args)
    actual = y[val_origins[:, None] + np.arange(PRED_LEN)]
    runs = []
    for seed in args.seeds:
        set_seed(seed)
        model = make_model(marks.shape[1], args)
        print(f"[{args.tag}] seed={seed} params={count_params(model)} "
              f"train_windows={len(train_origins)} val_windows={len(val_origins)}", flush=True)
        t0 = time.time()
        ran, best_epoch, history = train_epochs(model, windows, train_origins, args, seed,
                                                args.max_epochs, (val_origins, actual, scaler))
        forecast = predict(model, windows, val_origins, scaler)
        m = metrics(actual, forecast)
        m.update(seed=seed, params=count_params(model), epochs_run=ran, best_epoch=best_epoch,
                 fit_sec=time.time() - t0, history=history)
        runs.append(m)
        print(f"[{args.tag}] seed={seed} RMSE={m['RMSE']:.2f} MAE={m['MAE']:.2f} "
              f"sMAPE={m['sMAPE']:.2f} best_epoch={best_epoch}/{ran}", flush=True)
    summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
                   "std": float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0}
               for k in ["RMSE", "MAE", "sMAPE", "block_RMSE_mean", "epochs_run", "best_epoch",
                         "fit_sec"]}
    print(f"[{args.tag}] SUMMARY " + ", ".join(
        f"{k}={v['mean']:.2f}±{v['std']:.2f}" for k, v in summary.items()), flush=True)
    save(args, {"config": vars(args), "params": runs[0]["params"], "summary": summary,
                "runs": runs})


def run_final(args):
    """Early-stop on the validation split to pick the epoch count, optionally refit on the whole
    history for that many epochs, then forecast time_idx 43657..43824."""
    y, external = load_series()
    val_start, train_origins, val_origins = split_origins(args)
    actual = y[val_origins[:, None] + np.arange(PRED_LEN)]
    total_epochs, total_params, forecasts, log = 0, 0, [], []
    for seed in args.seeds:                     # >1 seed = an ensemble; P and E are summed
        set_seed(seed)
        scaler = TargetScaler(y[:val_start], args.target)
        marks, n_exog = build_marks(external, val_start, args)
        windows = Windows(scaler.scale(y), marks, n_exog, args)
        model = make_model(marks.shape[1], args)
        if args.fixed_epochs:
            ran, best_epoch = 0, args.fixed_epochs
        else:
            ran, best_epoch, _ = train_epochs(model, windows, train_origins, args, seed,
                                              args.max_epochs, (val_origins, actual, scaler))
        entry = {"seed": seed, "early_stop_epochs_run": ran, "best_epoch": best_epoch}
        if not args.fixed_epochs:
            entry["val"] = metrics(actual, predict(model, windows, val_origins, scaler))
        epochs = ran
        if args.refit:
            set_seed(seed)
            scaler = TargetScaler(y[:N_TRAIN], args.target)
            marks, n_exog = build_marks(external, N_TRAIN, args)
            windows = Windows(scaler.scale(y), marks, n_exog, args)
            model = make_model(marks.shape[1], args)
            all_origins = np.arange(args.seq_len, N_TRAIN - PRED_LEN + 1, args.stride)
            train_epochs(model, windows, all_origins, args, seed, best_epoch)
            epochs += best_epoch
        entry["epochs_counted"] = epochs
        forecasts.append(predict(model, windows, np.array([N_TRAIN]), scaler)[0])
        total_epochs += epochs
        total_params += count_params(model)
        log.append(entry)
        print(entry, flush=True)
    forecast = np.mean(forecasts, 0)
    assert forecast.shape == (PRED_LEN,) and np.isfinite(forecast).all()
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame({"time_idx": np.arange(N_TRAIN + 1, N_FULL + 1), "value": forecast}).to_csv(
        RESULTS / f"{args.tag}_forecast.csv", index=False)
    pasted = ", ".join(f"{v:.4f}" for v in forecast)
    (RESULTS / f"{args.tag}_submission.txt").write_text(pasted + "\n")
    save(args, {"config": vars(args), "P": total_params, "E": total_epochs, "seeds": log,
                "forecast": forecast.tolist()})
    print(f"\nP (trainable params) = {total_params}\nE (training epochs) = {total_epochs}")
    print(f"submission string written to results/{args.tag}_submission.txt")


def run_baselines(args):
    y, external = load_series()
    val_start, _, val_origins = split_origins(args)
    actual = y[val_origins[:, None] + np.arange(PRED_LEN)]
    hours = np.arange(PRED_LEN)
    rows = {}
    rows["train mean"] = np.full_like(actual, y[:val_start].mean())
    rows["persistence (last value)"] = np.repeat(y[val_origins - 1][:, None], PRED_LEN, 1)
    rows["seasonal naive (24)"] = y[val_origins[:, None] - 24 + hours % 24]
    rows["mean of last 7 days"] = np.repeat(
        np.array([y[o - 168:o].mean() for o in val_origins])[:, None], PRED_LEN, 1)
    for name, f in rows.items():
        print(f"{name:28s}", {k: round(v, 2) for k, v in metrics(actual, f).items()})
    save(args, {name: metrics(actual, f) for name, f in rows.items()})


def save(args, payload):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(payload, indent=1))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--exog", choices=["none", "past", "full"], default="full")
    p.add_argument("--features", default="ABCDEFGHIJ",
                   help="letters of external features to use (only when --exog != none)")
    p.add_argument("--no_daily_phase", dest="daily_phase", action="store_false")
    p.add_argument("--target", choices=["raw", "log"], default="raw")
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=64)
    p.add_argument("--e_layers", type=int, default=1)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--kernel", type=int, default=25)
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--onecycle", action="store_true")
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--val_len", type=int, default=24 * 7 * 26 + PRED_LEN,
                   help="final steps of the history used for validation targets (~6 months)")
    p.add_argument("--val_stride", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--fixed_epochs", type=int, default=0)
    args = p.parse_args()
    letters = set(args.features.upper())
    args.features_cont = [c for c in CONTINUOUS if c[-1] in letters]
    args.features_bin = [c for c in BINARY if c[-1] in letters]
    return args


if __name__ == "__main__":
    args = parse()
    if args.baselines:
        run_baselines(args)
    elif args.final:
        run_final(args)
    else:
        run_validation(args)
