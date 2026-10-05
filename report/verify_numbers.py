"""Check the numbers quoted in report/main.tex against the saved outputs.

Each check recomputes a value from the files in Question 1/results/design and
Question 2 - Leaderboard/results (or the data) and compares it with the value as written in the
report, allowing for the rounding shown. Run from the repository root:

    python report/verify_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
Q1 = ROOT / "Question 1" / "results" / "design"
Q2 = ROOT / "Question 2 - Leaderboard"
R2 = Q2 / "results"
sys.path.insert(0, str(Q2))

results = []


def check(label, reported, computed, decimals=None, tol=None):
    """reported: the value as printed; decimals: digits shown (sets the rounding tolerance)."""
    if tol is None:
        d = decimals if decimals is not None else (len(str(reported).split(".")[1]) if "." in str(reported) else 0)
        tol = 0.5 * 10 ** (-d) + 1e-9
    ok = abs(float(reported) - float(computed)) <= tol
    results.append((ok, label, reported, computed))


def csv(name):
    return pd.read_csv(Q1 / f"{name}.csv")


def j(name):
    return json.loads((R2 / f"{name}.json").read_text())


# ------------------------------------------------------------------ Task 1
t11 = csv("1.1").set_index("model")
cols = {"All": "established stations RMSE (m/s²)", "A": "established stations, Product A RMSE",
        "B": "established stations, Product B RMSE", "C": "established stations, Product C RMSE",
        "4All": "held-out station 4 RMSE (m/s²)", "4A": "held-out station 4, Product A RMSE",
        "4B": "held-out station 4, Product B RMSE", "4C": "held-out station 4, Product C RMSE"}
table11 = {
    "Shared ridge": [0.767, 0.805, 0.854, 0.621, 0.876, 0.909, 0.982, 0.713],
    "Sensor-specific ridge": [0.768, 0.804, 0.852, 0.630, None, None, None, None],
    "Period-routed ridge": [0.166, 0.181, 0.164, 0.152, 0.173, 0.184, 0.182, 0.150],
    "Raw Attention": [0.417, 0.396, 0.447, 0.405, 0.581, 0.604, 0.561, 0.576],
}
for model, vals in table11.items():
    for key, v in zip(cols, vals):
        if v is not None:
            check(f"1.1 {model} {key}", v, t11.loc[model, cols[key]], 3)
est, s4 = t11[cols["All"]], t11[cols["4All"]]
check("1A shared->routed reduction 78%", 78, 100 * (1 - est["Period-routed ridge"] / est["Shared ridge"]), 0)
check("1A routed vs attention 2.5x", 2.5, est["Raw Attention"] / est["Period-routed ridge"], 1)
check("1A Station-4 ratio 3.4x", 3.4, s4["Raw Attention"] / s4["Period-routed ridge"], 1)
check("1A routed degrades 4%", 4, 100 * (s4["Period-routed ridge"] / est["Period-routed ridge"] - 1), 0)
check("1A shared degrades 14%", 14, 100 * (s4["Shared ridge"] / est["Shared ridge"] - 1), 0)
check("1A attention degrades 39%", 39, 100 * (s4["Raw Attention"] / est["Raw Attention"] - 1), 0)
check("1A params shared 4.6k", 4.6, t11.loc["Shared ridge", "parameters"] / 1000, 1)
check("1A params routed 59.9k", 59.9, t11.loc["Period-routed ridge", "parameters"] / 1000, 1)
check("1A params attention 368k", 368, t11.loc["Raw Attention", "parameters"] / 1000, 0)
check("1A attention fit ~102 s", 102, t11.loc["Raw Attention", "fit seconds"], 0)

t12 = csv("1.2")
w = lambda prod, model: t12[(t12["product"] == prod) & (t12.model == model)]["window RMSE (m/s²)"].iloc[0]
check("1B Product A shared window RMSE 0.78", 0.78, w("Product A", "Shared ridge"))
check("1B Product A routed 0.23", 0.23, w("Product A", "Period-routed ridge"))
check("1B Product C shared 0.45", 0.45, w("Product C", "Shared ridge"))
check("1B Product C routed 0.15", 0.15, w("Product C", "Period-routed ridge"))
check("1B Product A period 16.1", 16.1, t12["actual period (samples)"].iloc[0])
check("1B Product C period 30.1", 30.1, t12[t12["product"] == "Product C"]["actual period (samples)"].iloc[0])
t13 = csv("1.3").iloc[-1]
check("1B max attention difference 0.43", 0.43, t13["max |Attention weight difference|"])
check("1B mean attention difference 0.013", 0.013, t13["mean |Attention weight difference|"])

t21 = csv("2.1").set_index("width")
check("2 residual std m=17 0.58", 0.58, t21.loc[17, "residual std (m/s²)"])
check("2 residual std m=41 1.57", 1.57, t21.loc[41, "residual std (m/s²)"])
check("2 selected width 41", 41, t21.index[t21.selected].item(), 0)
t22 = csv("2.2").set_index("representation")
for rep, v in [("raw", 68.7), ("width 17", 98.0), ("width 25", 98.9), ("width 33", 99.3), ("width 41", 99.8)]:
    check(f"2 oracle recovery {rep}", v, 100 * t22.loc[rep, "oracle recovery within one sample"], 1)
t23 = csv("2.3")
g = lambda m, pop, prod: t23[(t23.model == m) & (t23.population == pop) & (t23["product"] == prod)]["RMSE (m/s²)"].iloc[0]
rows23 = {"Raw Attention": [0.396, 0.447, 0.405, 0.604, 0.561, 0.576],
          "Attention + decomposition": [0.277, 0.339, 0.288, 0.397, 0.482, 0.466]}
keys23 = [(p, x) for p in ["established", "held-out"] for x in ["Product A", "Product B", "Product C"]]
for m, vals in rows23.items():
    for (p, x), v in zip(keys23, vals):
        check(f"2.3 {m} {p} {x}", v, g(m, p, x), 3)
for (p, x), v in zip(keys23, [-30, -24, -29, -34, -14, -19]):
    check(f"2.3 relative change {p} {x}", v, 100 * (g("Attention + decomposition", p, x) / g("Raw Attention", p, x) - 1), 0)

t32 = csv("3.2")
check("3 residual corr at 16 = 0.896", 0.896, t32["residual signal correlation"].iloc[0])
check("3 residual corr at 32 = 0.897", 0.897, t32["residual signal correlation"].iloc[1])
check("3 P = 16.1", 16.1, t32["expected delay (samples)"].iloc[0])
check("3 2P = 32.3", 32.3, t32["expected delay (samples)"].iloc[1])
t31 = csv("3.1")
for i, v in enumerate([4, -4, 4, -4]):
    check(f"3.1 delay score tau={i}", v, t31["delay_scores (FFT)"].iloc[i], 0)

t41 = csv("4.1").set_index("model")
mse_e, mse_4 = "established stations MSE (m/s²)²", "held-out station 4 MSE (m/s²)²"
table4 = {  # RMSE est, MSE est, RMSE st4, MSE st4, params, fit s
    "Raw Attention": [0.417, 0.174, 0.581, 0.338, 368368, 101.7],
    "Attention + decomposition": [0.302, 0.091, 0.450, 0.202, 373024, 188.3],
    "Raw delay mixer": [0.230, 0.053, 0.346, 0.119, 368370, 305.5],
    "Autoformer-inspired": [0.196, 0.038, 0.245, 0.060, 373026, 142.7],
}
for m, v in table4.items():
    r = t41.loc[m]
    for lab, rep, comp, d in zip(["RMSE est", "MSE est", "RMSE st4", "MSE st4", "params", "fit s"], v,
                                 [r[cols["All"]], r[mse_e], r[cols["4All"]], r[mse_4], r["parameters"], r["fit seconds"]],
                                 [3, 3, 3, 3, 0, 1]):
        check(f"4.1 {m} {lab}", rep, comp, d)
for m, v in {"Shared ridge": [0.588, 0.767, 4608, 0.003], "Sensor-specific ridge": [0.590, None, 13824, 0.029],
             "Period-routed ridge": [0.028, 0.030, 59904, 0.005]}.items():
    r = t11.loc[m]
    check(f"4.1 {m} MSE est", v[0], r[mse_e], 3)
    if v[1] is not None:
        check(f"4.1 {m} MSE st4", v[1], r[mse_4], 3)
    check(f"4.1 {m} params", v[2], r["parameters"], 0)
    check(f"4.1 {m} fit s", v[3], r["fit seconds"], 3)
base = t41.loc["Raw Attention", mse_e]
for m, d, pct in [("Attention + decomposition", 0.082, 47), ("Raw delay mixer", 0.120, 69), ("Autoformer-inspired", 0.135, 78)]:
    check(f"4b MSE gain {m}", d, base - t41.loc[m, mse_e], 3)
    check(f"4b MSE gain % {m}", pct, 100 * (base - t41.loc[m, mse_e]) / base, 0)
sum_gain = (base - t41.loc["Attention + decomposition", mse_e]) + (base - t41.loc["Raw delay mixer", mse_e])
check("4b sum of individual gains 0.203", 0.203, sum_gain, 3)
check("4c routed vs Autoformer params 6x", 6, t41.loc["Autoformer-inspired", "parameters"] / t11.loc["Period-routed ridge", "parameters"], 0)

t42 = csv("4.2")
hz = lambda m, pop, h: t42[(t42.model == m) & (t42.population == pop) & (t42.horizon == h)]["RMSE (m/s²)"].iloc[0]
for m, a, b in [("Raw Attention", 0.144, 0.488), ("Attention + decomposition", 0.149, 0.421),
                ("Raw delay mixer", 0.119, 0.373), ("Autoformer-inspired", 0.132, 0.242), ("Period-routed ridge", 0.112, 0.228)]:
    check(f"4.2 {m} h=1", a, hz(m, "established", 1), 3)
    check(f"4.2 {m} h=48", b, hz(m, "established", 48), 3)
check("4.2 Raw Attention growth 3.4x", 3.4, hz("Raw Attention", "established", 48) / hz("Raw Attention", "established", 1), 1)
check("4.2 Autoformer growth 1.8x", 1.8, hz("Autoformer-inspired", "established", 48) / hz("Autoformer-inspired", "established", 1), 1)
check("4.2 St4 Raw Attention h=48 0.768", 0.768, hz("Raw Attention", "held-out", 48), 3)
check("4.2 St4 routed h=48 0.212", 0.212, hz("Period-routed ridge", "held-out", 48), 3)

t43 = csv("4.3").set_index("population")
for pop, v in {"established": [0.0275, 0.166, 0.0260, 0.161], "held-out": [0.0298, 0.173, 0.0333, 0.183]}.items():
    for lab, rep, col, d in zip(["val MSE", "val RMSE", "test MSE", "test RMSE"], v,
                                ["validation MSE", "validation RMSE (m/s²)", "test MSE", "test RMSE (m/s²)"], [4, 3, 4, 3]):
        check(f"4.3 {pop} {lab}", rep, t43.loc[pop, col], d)
gap = max(abs(t43.loc[p, "test RMSE (m/s²)"] / t43.loc[p, "validation RMSE (m/s²)"] - 1) for p in t43.index)
check("4.3 test within ~6% of validation", 6, 100 * gap, tol=1.0)

# ------------------------------------------------------------------ Task 2
import pipeline as P  # noqa: E402

y, ext = P.load_series()
yt = y[:P.N_TRAIN]
for q, v in [(0.5, 73), (0.99, 418)]:
    check(f"2.1 quantile {q}", v, np.quantile(yt, q), 0)
check("2.1 max 994", 994, yt.max(), 0)
yc = yt - yt.mean()
f = np.fft.rfft(yc, n=2 * len(yc))
acf = np.fft.irfft(f * np.conj(f))[:len(yc)]
acf /= acf[0]
for lag, v in [(1, 0.97), (12, 0.57), (24, 0.40), (48, 0.16), (96, 0.02), (168, 0.03)]:
    check(f"2.1 ACF lag {lag}", v, acf[lag], 2)
ly = np.log1p(yt)
check("2.1 corr D vs log y", -0.35, np.corrcoef(ly, ext.feature_D.values[:P.N_TRAIN])[0, 1], 2)
check("2.1 corr H vs log y", -0.35, np.corrcoef(ly, ext.feature_H.values[:P.N_TRAIN])[0, 1], 2)
check("2.1 mean y when H active 70", 70, yt[ext.feature_H.values[:P.N_TRAIN] == 1].mean(), 0)
check("2.1 mean y when J active 125", 125, yt[ext.feature_J.values[:P.N_TRAIN] == 1].mean(), 0)

cfg = j("ablation_full")["config"]
val_start = P.N_TRAIN - cfg["val_len"]
check("2.2 training targets end at 39,120", 39120, val_start, 0)
check("2.2 validation length 4,536", 4536, cfg["val_len"], 0)
check("2.2 183 validation blocks", 183, len(range(val_start, P.N_TRAIN - P.PRED_LEN + 1, cfg["val_stride"])), 0)
b = j("baselines")
for name, v in [("train mean", 87.8), ("mean of last 7 days", 94.7), ("seasonal naive (24)", 117.9), ("persistence (last value)", 122.2)]:
    check(f"2.2 baseline {name}", v, b[name]["RMSE"], 1)
check("2.4 baseline MAE 67.9", 67.9, b["train mean"]["MAE"], 1)
check("2.4 baseline sMAPE 79.4", 79.4, b["train mean"]["sMAPE"], 1)
check("2.4 baseline block RMSE 80.9", 80.9, b["train mean"]["block_RMSE_mean"], 1)
check("2.3 params 22,081", 22081, j("ablation_full")["params"], 0)

for tag, rm, rs, mae, ms, sm, ss, blk in [("ablation_none", 89.64, 0.74, 69.1, 1.1, 79.5, 0.3, 81.8),
                                         ("ablation_past", 88.91, 0.44, 69.3, 1.3, 79.5, 0.5, 81.7),
                                         ("ablation_full", 71.13, 1.46, 55.9, 2.9, 70.9, 1.6, 67.3)]:
    s = j(tag)["summary"]
    check(f"2.4 {tag} RMSE", rm, s["RMSE"]["mean"], 2)
    check(f"2.4 {tag} RMSE std", rs, s["RMSE"]["std"], 2)
    check(f"2.4 {tag} MAE", mae, s["MAE"]["mean"], 1)
    check(f"2.4 {tag} MAE std", ms, s["MAE"]["std"], 1)
    check(f"2.4 {tag} sMAPE", sm, s["sMAPE"]["mean"], 1)
    check(f"2.4 {tag} sMAPE std", ss, s["sMAPE"]["std"], 1)
    check(f"2.4 {tag} block RMSE", blk, s["block_RMSE_mean"]["mean"], 1)
none, full = j("ablation_none")["summary"], j("ablation_full")["summary"]
check("2.4 RMSE drop 18.5", 18.5, none["RMSE"]["mean"] - full["RMSE"]["mean"], 1)
check("2.4 RMSE drop ~21%", 21, 100 * (1 - full["RMSE"]["mean"] / none["RMSE"]["mean"]), 0)
check("2.4 sMAPE drop ~11%", 11, 100 * (1 - full["sMAPE"]["mean"] / none["sMAPE"]["mean"]), 0)
check("2.4 past-only change 0.7", 0.7, none["RMSE"]["mean"] - j("ablation_past")["summary"]["RMSE"]["mean"], 1)

tune = [("tune_log", 66.80, 0.89, 44.1, 54.5, 5.7), ("tune_lr3e-4", 67.60, 1.85, 51.5, 67.5, 6.0),
        ("tune_d16", 67.89, 2.36, 51.6, 68.4, 4.3), ("tune_huber", 69.92, 2.07, 53.1, 68.5, 3.0),
        ("tune_noC", 70.17, 0.66, 55.5, 70.5, 3.7), ("tune_seq168", 70.20, 6.44, 53.7, 69.9, 3.3),
        ("tune_drop03", 71.01, 0.89, 55.6, 70.1, 4.7), ("tune_seq96", 76.25, 3.70, 57.4, 73.2, 3.3),
        ("combined", 68.54, 1.30, 44.6, 54.8, 10.0), ("log_d16", 67.41, 2.96, 43.7, 54.1, 8.0),
        ("ablation_full", 71.13, 1.46, 55.9, 70.9, 3.0)]
for tag, rm, rs, mae, sm, ep in tune:
    s = j(tag)["summary"]
    check(f"2.5 {tag} RMSE", rm, s["RMSE"]["mean"], 2)
    check(f"2.5 {tag} RMSE std", rs, s["RMSE"]["std"], 2)
    check(f"2.5 {tag} MAE", mae, s["MAE"]["mean"], 1)
    check(f"2.5 {tag} sMAPE", sm, s["sMAPE"]["mean"], 1)
    check(f"2.5 {tag} epochs run", ep, s["epochs_run"]["mean"], 1)
for tag, prm in [("tune_noC", 22017), ("combined", 5921), ("log_d16", 5921)]:
    check(f"2.5 {tag} params", prm, j(tag)["params"], 0)
check("2.5 tune_d16 params ~5.9k", 5.9, j("tune_d16")["params"] / 1000, 1)
lg = j("tune_log")["summary"]
check("2.5 log RMSE gain 4.3", 4.3, full["RMSE"]["mean"] - lg["RMSE"]["mean"], 1)
check("2.5 log MAE gain 21%", 21, 100 * (1 - lg["MAE"]["mean"] / full["MAE"]["mean"]), 0)
check("2.5 log sMAPE gain 23%", 23, 100 * (1 - lg["sMAPE"]["mean"] / full["sMAPE"]["mean"]), 0)

fin = j("final")
check("2.6 #1 P", 22081, fin["P"], 0)
check("2.6 #1 E", 12, fin["E"], 0)
s0 = fin["seeds"][0]
check("2.6 #1 early-stop epochs 7", 7, s0["early_stop_epochs_run"], 0)
check("2.6 #1 best epoch 5", 5, s0["best_epoch"], 0)
check("2.6 #1 val RMSE 67.46", 67.46, s0["val"]["RMSE"], 2)
check("2.6 #1 forecast median 23.6", 23.6, np.median(fin["forecast"]), 1)
check("2.6 val block RMSE median ~49", 49, s0["val"]["block_RMSE_median"], 0)

M = R2 / "members"
mem = {p.stem: np.load(p) for p in sorted(M.glob("*.npz"))}
actual = mem["log_s0"]["actual"]
block = lambda fc: np.sqrt(((fc - actual) ** 2).mean(1))
stab = np.load(R2 / "stability_forecasts.npy")
check("2.6 stability min mean 26", 26, stab.mean(1).min(), 0)
check("2.6 stability max mean 71", 71, stab.mean(1).max(), 0)
check("2.6 pairwise RMSE 28.6", 28.6, np.mean([np.sqrt(((a - c) ** 2).mean()) for i, a in enumerate(stab) for c in stab[i + 1:]]), 1)
check("2.6 submitted model = lowest of six", 1, float(np.argmin(stab.mean(1)) == 1), 0)
lvl = actual.mean(1)
hi = lvl > np.quantile(lvl, 0.75)
check("2.6 log s0 bias top quartile -33.7", -33.7, (mem["log_s0"]["val"] - actual)[hi].mean(), 1)

singles = [P.metrics(actual, mem[f"log_s{i}"]["val"]) for i in range(3)]
for i, v in enumerate([67.46, 65.80, 67.15]):
    check(f"2.6 #2 single seed {i} RMSE", v, singles[i]["RMSE"], 2)
check("2.6 #2 single MAE low 43.5", 43.5, min(s["MAE"] for s in singles), 1)
check("2.6 #2 single MAE high 44.4", 44.4, max(s["MAE"] for s in singles), 1)
p90 = [np.quantile(block(mem[f"log_s{i}"]["val"]), 0.9) for i in range(3)]
p97 = [np.quantile(block(mem[f"log_s{i}"]["val"]), 0.97) for i in range(3)]
check("2.6 #2 single p90 low 100", 100, min(p90), 0)
check("2.6 #2 single p90 high 106", 106, max(p90), 0)
check("2.6 #2 single p97 low 127", 127, min(p97), 0)
check("2.6 #2 single p97 high 141", 141, max(p97), 0)
ens = np.mean([mem[f"log_s{i}"]["val"] for i in range(3)], 0)
m = P.metrics(actual, ens)
check("2.6 #2 ensemble RMSE 63.95", 63.95, m["RMSE"], 2)
check("2.6 #2 ensemble MAE 42.5", 42.5, m["MAE"], 1)
check("2.6 #2 ensemble p90 96", 96, np.quantile(block(ens), 0.9), 0)
check("2.6 #2 ensemble p97 126", 126, np.quantile(block(ens), 0.97), 0)
fe = j("final_ensemble")
check("2.6 #2 P 66,243", 66243, fe["P"], 0)
check("2.6 #2 E 17", 17, fe["E"], 0)
for i, v in enumerate([7, 6, 4]):
    check(f"2.6 #2 member {i} epochs", v, fe["seeds"][i]["early_stop_epochs_run"], 0)

ee = j("ensemble_eval")
rows = {"log x3 (current submission)": (63.95, None, None, None, 126, -36.8, 66243, 17),
        "log x5": (63.39, -0.55, -1.39, 0.22, 127, -38.8, 110405, 30),
        "raw x3": (69.16, 5.21, 1.31, 9.27, 114, -18.1, 66243, 9),
        "raw x5": (67.66, 3.71, -0.21, 7.97, 113, -13.5, 110405, 17),
        "log x5 + raw x5": (62.44, -1.51, -3.65, 0.64, 117, -26.2, 220810, 47)}
for name, (rm, dlt, lo, hi_, p97v, bias, prm, ep) in rows.items():
    r = ee[name]
    check(f"2.6 #3 {name} RMSE", rm, r["RMSE"], 2)
    if dlt is not None:
        check(f"2.6 #3 {name} delta", dlt, r["diff"], 2)
        check(f"2.6 #3 {name} CI low", lo, r["ci"][0], 2)
        check(f"2.6 #3 {name} CI high", hi_, r["ci"][1], 2)
    check(f"2.6 #3 {name} p97", p97v, r["p97"], 0)
    check(f"2.6 #3 {name} bias", bias, r["bias_high"], 1)
    check(f"2.6 #3 {name} P", prm, r["P"], 0)
    check(f"2.6 #3 {name} E", ep, r["E"], 0)
mx = j("mix_w03")
check("2.6 #3 weighted mix RMSE 62.05", 62.05, mx["val"]["RMSE"], 2)
check("2.6 #3 P 220,810", 220810, mx["P"], 0)
check("2.6 #3 E 47", 47, mx["E"], 0)
check("2.6 #3 log members epochs 30", 30, sum(v for k, v in mx["epochs_per_member"].items() if k.startswith("log")), 0)
check("2.6 #3 raw members epochs 17", 17, sum(v for k, v in mx["epochs_per_member"].items() if k.startswith("raw")), 0)
check("2.6 #3 validation gain ~1.9", 1.9, 63.95 - mx["val"]["RMSE"], 1)
logm = np.mean([mem[f"log_s{i}"]["val"] for i in range(5)], 0)
rawm = np.mean([mem[f"raw_s{i}"]["val"] for i in range(5)], 0)
eq = P.metrics(actual, 0.5 * logm + 0.5 * rawm)
check("2.6 #3 w=0.3 beats equal mix on MAE", 1, float(mx["val"]["MAE"] < eq["MAE"]), 0)
check("2.6 #3 w=0.3 beats equal mix on sMAPE", 1, float(mx["val"]["sMAPE"] < eq["sMAPE"]), 0)

# Leaderboard scores (from the leaderboard page; checked for internal consistency only)
for att, rmse, score in [(1, 142.0354, 142.0414), (2, 112.1470, 112.1557), (3, 102.2065, 102.2305)]:
    check(f"2.6 leaderboard #{att} penalty", {1: 0.006, 2: 0.009, 3: 0.024}[att], score - rmse, 3)

# ------------------------------------------------------------------ report
bad = [r for r in results if not r[0]]
for ok, label, rep, comp in results:
    if not ok:
        print(f"MISMATCH  {label}: report {rep} vs computed {comp:.4f}")
print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
