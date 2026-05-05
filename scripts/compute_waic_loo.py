"""WAIC + PSIS-LOO computation from saved log_lik.

Inputs : state.npz files containing key 'log_lik' shape (S, N_obs).
Output : Markdown comparison table + npz with raw values.

WAIC formula (Watanabe 2010, Vehtari 2017 eq. 14-15):
    lpd_i        = log( mean_s exp(log_lik[s, i]) )
    p_waic_i     = var_s log_lik[s, i]
    elpd_waic_i  = lpd_i - p_waic_i
    WAIC         = -2 * sum_i elpd_waic_i

PSIS-LOO (Vehtari 2017): Pareto-smoothed importance sampling.
    r_si  = 1 / p(y_i | theta_s) ∝ exp(-log_lik[s, i])
    Smooth top 20% tail with generalized Pareto.
    elpd_loo_i = log( sum_s w_si * p(y_i | theta_s) / sum_s w_si )
    LOO        = -2 * sum_i elpd_loo_i
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def waic_from_loglik(log_lik: np.ndarray, mask: np.ndarray | None = None):
    """Compute WAIC. log_lik shape (S, N). mask optional (N,) — exclude rows."""
    if mask is not None:
        log_lik = log_lik[:, mask.astype(bool)]
    S, N = log_lik.shape
    # log mean exp over S, per i
    lse = np.logaddexp.reduce(log_lik, axis=0) - np.log(S)   # (N,)
    p_waic = log_lik.var(axis=0)                             # (N,)
    elpd_i = lse - p_waic                                    # (N,)
    elpd_waic = float(elpd_i.sum())
    se = float(np.sqrt(N) * elpd_i.std(ddof=1))
    waic = -2.0 * elpd_waic
    waic_se = 2.0 * se
    return {
        "elpd_waic": elpd_waic, "se": se,
        "waic": waic, "waic_se": waic_se,
        "p_waic_total": float(p_waic.sum()),
        "n_obs": N,
    }


def psis_loo_from_loglik(log_lik: np.ndarray, mask: np.ndarray | None = None):
    """PSIS-LOO via Pareto-smoothed importance sampling (lightweight)."""
    if mask is not None:
        log_lik = log_lik[:, mask.astype(bool)]
    S, N = log_lik.shape
    lw = -log_lik                                             # (S, N)
    lw -= np.max(lw, axis=0, keepdims=True)                   # log-stabilize
    # Pareto-smooth top-k tail per column
    k_tail = max(int(np.ceil(min(0.2 * S, 3 * np.sqrt(S)))), 5)
    elpd_i = np.empty(N)
    pareto_k = np.empty(N)
    for i in range(N):
        ll_i = log_lik[:, i]
        lw_i = lw[:, i]
        # Sort weights desc, take top k for tail fit
        order = np.argsort(lw_i)[::-1]
        tail_idx = order[:k_tail]
        tail_lw = lw_i[tail_idx]
        # Fit GPD to tail (Vehtari 2017 §3.2 simplified MLE)
        cutoff = tail_lw[-1]
        excess = np.exp(tail_lw - cutoff) - 1.0
        excess = np.maximum(excess, 1e-12)
        # Method-of-moments GPD fit
        m, v = excess.mean(), excess.var() + 1e-12
        khat = 0.5 * (1.0 - m * m / v)
        sigma = m * (1.0 - khat)
        pareto_k[i] = khat
        # Replace tail weights with smoothed quantiles
        if khat < 1.0 and sigma > 0:
            order_tail = np.argsort(tail_lw)[::-1]
            ranks = (np.arange(k_tail) + 0.5) / k_tail
            if abs(khat) < 1e-6:
                smoothed_excess = -sigma * np.log(1.0 - ranks)
            else:
                smoothed_excess = (sigma / khat) * ((1.0 - ranks) ** (-khat) - 1.0)
            smoothed_lw = cutoff + np.log1p(smoothed_excess)
            lw_i = lw_i.copy()
            lw_i[tail_idx[order_tail]] = smoothed_lw
        # Cap weights at log S (Vehtari 2017 §3.4)
        lw_i = np.minimum(lw_i, np.log(S))
        # elpd_i = log( sum_s exp(lw_si + log_lik_si) / sum_s exp(lw_si) )
        lw_max = lw_i.max()
        num = np.exp(lw_i - lw_max + ll_i).sum()
        den = np.exp(lw_i - lw_max).sum()
        elpd_i[i] = np.log(num / den)
    elpd_loo = float(elpd_i.sum())
    se = float(np.sqrt(N) * elpd_i.std(ddof=1))
    loo = -2.0 * elpd_loo
    return {
        "elpd_loo": elpd_loo, "se": se,
        "loo": loo, "loo_se": 2.0 * se,
        "pareto_k_max": float(pareto_k.max()),
        "pareto_k_p99": float(np.quantile(pareto_k, 0.99)),
        "pareto_k_bad_count": int((pareto_k > 0.7).sum()),
        "n_obs": N,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-files", nargs="+", required=True, type=Path,
                        help="List of state.npz with log_lik")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels (one per state file)")
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()
    assert len(args.state_files) == len(args.labels), "labels must match state files"

    rows = []
    raw = {}
    for path, label in zip(args.state_files, args.labels):
        z = np.load(path, allow_pickle=True)
        if "log_lik" not in z.files:
            print(f"[skip] {label}: no log_lik in {path}")
            continue
        ll = z["log_lik"]
        # Filter zero log_lik (mask=0 rows, where Bernoulli * 0 = 0)
        nonzero = ~(np.abs(ll).sum(axis=0) == 0)
        ll_active = ll[:, nonzero]
        print(f"=== {label} ===   log_lik shape: {ll.shape} → active {ll_active.shape}")
        waic = waic_from_loglik(ll, mask=nonzero)
        loo = psis_loo_from_loglik(ll, mask=nonzero)
        rows.append({"label": label, "waic": waic, "loo": loo})
        raw[f"{label}_waic"] = waic
        raw[f"{label}_loo"] = loo
        print(f"  WAIC  = {waic['waic']:.1f} ± {waic['waic_se']:.1f}, "
              f"elpd_waic = {waic['elpd_waic']:.1f}, p_waic = {waic['p_waic_total']:.1f}")
        print(f"  LOO   = {loo['loo']:.1f} ± {loo['loo_se']:.1f}, "
              f"elpd_loo = {loo['elpd_loo']:.1f}, "
              f"pareto-k max = {loo['pareto_k_max']:.2f} (bad={loo['pareto_k_bad_count']})")

    # Comparison table
    md = ["# WAIC + PSIS-LOO comparison", ""]
    md.append("| Method | N_obs | WAIC | WAIC SE | LOO | LOO SE | p_waic | Pareto-k max | bad k>0.7 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        w, l = r["waic"], r["loo"]
        md.append(
            f"| {r['label']} | {w['n_obs']} | "
            f"{w['waic']:.1f} | {w['waic_se']:.1f} | "
            f"{l['loo']:.1f} | {l['loo_se']:.1f} | "
            f"{w['p_waic_total']:.1f} | "
            f"{l['pareto_k_max']:.2f} | {l['pareto_k_bad_count']} |"
        )
    md.append("")
    if len(rows) >= 2:
        md.append("## Pairwise ELPD difference (positive favors first label)")
        md.append("| A vs B | ΔELPD-WAIC | ΔELPD-LOO |")
        md.append("|---|---|---|")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                d_waic = a["waic"]["elpd_waic"] - b["waic"]["elpd_waic"]
                d_loo = a["loo"]["elpd_loo"] - b["loo"]["elpd_loo"]
                md.append(f"| {a['label']} vs {b['label']} | {d_waic:+.1f} | {d_loo:+.1f} |")
    md.append("")
    md.append("Notes: ΔELPD > 4 (with SE not overlapping zero) is conventionally "
              "considered model preference. Pareto-k > 0.7 indicates unstable "
              "PSIS-LOO estimates for those observations.")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    np.savez(args.out_md.with_suffix(".npz"), **{k: str(v) for k, v in raw.items()})
    print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
