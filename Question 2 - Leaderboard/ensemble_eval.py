"""Score ensembles of saved members (results/members/*.npz) on the validation blocks.

Each ensemble is compared with the current leaderboard submission (log target, seeds 0-2) by a
paired bootstrap. Validation origins are 24 steps apart while targets span 168, so neighbouring
blocks overlap; resampling groups of 7 consecutive origins (one week) keeps dependent blocks together.
"""
import json

import numpy as np

import pipeline as P

MEMBERS = P.RESULTS / "members"
REFERENCE = ["log_s0", "log_s1", "log_s2"]
ENSEMBLES = {
    "log x3 (current submission)": REFERENCE,
    "log x5": [f"log_s{i}" for i in range(5)],
    "raw x3": [f"raw_s{i}" for i in range(3)],
    "raw x5": [f"raw_s{i}" for i in range(5)],
    "log x3 + raw x3": REFERENCE + [f"raw_s{i}" for i in range(3)],
    "log x5 + raw x5": [f"log_s{i}" for i in range(5)] + [f"raw_s{i}" for i in range(5)],
}


def load(name):
    return dict(np.load(MEMBERS / f"{name}.npz"))


def block_sq(forecast, actual):
    return ((forecast - actual) ** 2).mean(1)            # per-block MSE


def main():
    members = {p.stem: load(p.stem) for p in sorted(MEMBERS.glob("*.npz"))}
    actual = next(iter(members.values()))["actual"]
    n_blocks = len(actual)
    groups = np.arange(n_blocks) // 7
    n_groups = groups.max() + 1
    rng = np.random.default_rng(0)
    draws = rng.integers(0, n_groups, size=(2000, n_groups))
    level = actual.mean(1)
    high = level > np.quantile(level, 0.75)

    print("single members:")
    for name, m in members.items():
        r = P.metrics(actual, m["val"])
        print(f"  {name:7s} RMSE {r['RMSE']:6.2f} MAE {r['MAE']:6.2f} epochs {int(m['epochs'])}")

    ref = np.mean([members[n]["val"] for n in REFERENCE], 0)
    ref_sq = block_sq(ref, actual)
    rows = {}
    print("\nensembles (diff = ensemble RMSE - current; 90% paired bootstrap CI):")
    for label, names in ENSEMBLES.items():
        if not all(n in members for n in names):
            continue
        f = np.mean([members[n]["val"] for n in names], 0)
        sq = block_sq(f, actual)
        r = P.metrics(actual, f)
        diffs = []
        for d in draws:
            idx = np.concatenate([np.flatnonzero(groups == g) for g in d])
            diffs.append(np.sqrt(sq[idx].mean()) - np.sqrt(ref_sq[idx].mean()))
        lo, hi = np.quantile(diffs, [0.05, 0.95])
        rm = np.sqrt(sq)
        bias_high = (f - actual)[high].mean()
        P_total = int(sum(members[n]["params"] for n in names))
        E_total = int(sum(members[n]["epochs"] for n in names))
        hidden = np.mean([members[n]["hidden"] for n in names], 0)
        rows[label] = dict(RMSE=r["RMSE"], MAE=r["MAE"], sMAPE=r["sMAPE"],
                           diff=float(r["RMSE"] - np.sqrt(ref_sq.mean())), ci=[float(lo), float(hi)],
                           p90=float(np.quantile(rm, .9)), p97=float(np.quantile(rm, .97)),
                           bias_high=float(bias_high), P=P_total, E=E_total,
                           hidden_mean=float(hidden.mean()), members=names)
        print(f"  {label:28s} RMSE {r['RMSE']:6.2f} MAE {r['MAE']:5.1f} sMAPE {r['sMAPE']:5.1f} | "
              f"diff {rows[label]['diff']:+5.2f} [{lo:+5.2f},{hi:+5.2f}] | p90/p97 "
              f"{rows[label]['p90']:4.0f}/{rows[label]['p97']:4.0f} | bias(top-q) {bias_high:+6.1f} | "
              f"P {P_total} E {E_total} | hidden mean {hidden.mean():5.1f}")
    (P.RESULTS / "ensemble_eval.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
