"""MIMIC-IV ARDS cohort loader (v3.1 BigQuery extraction, 28-day endpoint)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor


CCI_COLS = [
    "cci_mi", "cci_chf", "cci_pvd", "cci_cvd", "cci_dementia",
    "cci_cpd", "cci_rheum", "cci_pud", "cci_liver_mild", "cci_liver_severe",
    "cci_dm", "cci_dm_complications", "cci_paraplegia", "cci_renal",
    "cci_cancer", "cci_metastatic", "cci_aids",
]
CCI_WEIGHTS = {
    "cci_mi": 1, "cci_chf": 1, "cci_pvd": 1, "cci_cvd": 1, "cci_dementia": 1,
    "cci_cpd": 1, "cci_rheum": 1, "cci_pud": 1, "cci_liver_mild": 1,
    "cci_liver_severe": 3, "cci_dm": 1, "cci_dm_complications": 2,
    "cci_paraplegia": 2, "cci_renal": 2, "cci_cancer": 2,
    "cci_metastatic": 6, "cci_aids": 6,
}

# Time-varying clinical covariates fed as transition drivers and outcome covariates.
# Excludes algebraic components of MP (peep, ppeak, pplat, rr, tidvol_obs, driving_pressure,
# compliance) to avoid blocking the MP -> lung mechanics -> outcome causal pathway.
TV_COLS = [
    "pf_ratio", "paco2", "lactate", "map_mmhg",
    "heart_rate", "gcs_total", "creatinine", "temperature_c",
]
STATIC_COLS = ["anchor_age", "gender_M", "bmi_imputed", "charlson_index"]


@dataclass
class ARDSConfig:
    """Cohort + preprocessing hyperparameters.

    max_t = 28 follows the ARDS Network / LUNG SAFE 28-day mortality convention.
    exclude_tv_cols / exclude_static_cols enable LOCO sensitivity by dropping
    named features before standardization.
    """
    csv_path: Path
    n_bins: int = 20
    max_t: int = 28
    bin_sd_range: float = 2.5
    severity: Optional[str] = None
    impute_strategy: str = "ffill_then_zero"
    exclude_tv_cols: tuple[str, ...] = ()
    exclude_static_cols: tuple[str, ...] = ()


@dataclass
class ARDSCohort:
    """Tensors ready for VEM-SSM consumption.

    Shapes: Y (N,T,1), A_bin (N,T,K), L_dyn (N,T,p_dyn), C_static (N,p_static),
            at_risk (N,T,1), drivers (N,T,K+p_dyn+p_static),
            covariates (N,T,K+p_dyn+p_static+1).
    drivers and covariates differ only by the appended t_norm column.
    """
    Y: Tensor
    A_bin: Tensor
    L_dyn: Tensor
    C_static: Tensor
    at_risk: Tensor
    t_norm: Tensor
    drivers: Tensor
    covariates: Tensor
    bin_edges_mp: np.ndarray
    stay_ids: np.ndarray
    subject_ids: np.ndarray
    severity_label: np.ndarray
    feature_layout: dict


def _compute_charlson(df: pd.DataFrame) -> pd.Series:
    available = [c for c in CCI_COLS if c in df.columns]
    return sum(df[c].fillna(0) * CCI_WEIGHTS[c] for c in available)


def _classify_severity(pf: float) -> str:
    if pd.isna(pf):
        return "unknown"
    if pf < 100:
        return "severe"
    if pf < 200:
        return "moderate"
    return "mild"


def _build_mp_bin_edges(log_mp: np.ndarray, n_bins: int, sd_range: float):
    mu = float(np.nanmean(log_mp))
    sd = float(np.nanstd(log_mp))
    z_edges = np.linspace(-sd_range, sd_range, n_bins + 1)
    return np.exp(mu + sd * z_edges), mu, sd


def _assign_bins(mp: np.ndarray, edges: np.ndarray) -> np.ndarray:
    K = len(edges) - 1
    return np.clip(np.digitize(mp, edges, right=True) - 1, 0, K - 1)


def _build_at_risk_and_outcome(
    df_long: pd.DataFrame, max_t: int, death_col: str = "death_event",
) -> tuple[np.ndarray, np.ndarray]:
    if death_col not in df_long.columns:
        raise KeyError(f"`{death_col}` not found in long frame.")
    stays = df_long["stay_id"].drop_duplicates().to_numpy()
    N = len(stays)
    stay_to_idx = {sid: i for i, sid in enumerate(stays)}

    Y = np.zeros((N, max_t), dtype=np.float32)
    observed = np.zeros((N, max_t), dtype=np.float32)
    death_day = np.full(N, max_t, dtype=np.int32)

    sub = df_long[df_long["day_num"] < max_t]
    rows_idx = sub["stay_id"].map(stay_to_idx).to_numpy()
    days = sub["day_num"].to_numpy().astype(int)
    deaths = sub[death_col].fillna(0).to_numpy().astype(int)

    if {"pf_ratio", "peep", "mp_j_min"}.issubset(sub.columns):
        has_signal = (
            sub["pf_ratio"].notna() | sub["peep"].notna() | sub["mp_j_min"].notna()
        ).to_numpy()
        observed[rows_idx[has_signal], days[has_signal]] = 1.0
    else:
        observed[rows_idx, days] = 1.0

    death_mask = deaths == 1
    if death_mask.any():
        d_idx = rows_idx[death_mask]
        d_day = days[death_mask]
        order = np.argsort(d_day)
        seen = np.zeros(N, dtype=bool)
        for sub_i, day_t in zip(d_idx[order], d_day[order]):
            if not seen[sub_i]:
                death_day[sub_i] = day_t
                Y[sub_i, day_t] = 1.0
                seen[sub_i] = True

    t_grid = np.arange(max_t)[None, :]
    at_risk_within = (t_grid <= death_day[:, None]).astype(np.float32)
    return Y, at_risk_within * observed


def load_ards_cohort(cfg: ARDSConfig) -> ARDSCohort:
    """End-to-end loader producing tensors ready for model consumption."""
    df = pd.read_csv(cfg.csv_path)
    if "charlson_index" not in df.columns:
        cci_cols_present = [c for c in CCI_COLS if c in df.columns]
        df["charlson_index"] = (
            sum(df[c].fillna(0) * CCI_WEIGHTS[c] for c in cci_cols_present)
            if cci_cols_present else 0.0
        )

    tv_cols = [c for c in TV_COLS if c not in cfg.exclude_tv_cols]
    static_cols = [c for c in STATIC_COLS if c not in cfg.exclude_static_cols]

    first_day = (
        df.sort_values(["stay_id", "day_num"]).groupby("stay_id").first().reset_index()
    )
    if "severity" in df.columns:
        sev_map = first_day.set_index("stay_id")["severity"]
    else:
        sev_map = first_day.set_index("stay_id")["pf_ratio"].apply(_classify_severity)
    if cfg.severity is not None:
        keep = sev_map[sev_map == cfg.severity].index
        df = df[df["stay_id"].isin(keep)].copy()
        sev_map = sev_map.loc[keep]

    df = df[df["day_num"] < cfg.max_t].copy()
    df["charlson_index"] = _compute_charlson(df)
    if "gender" in df.columns:
        df["gender_M"] = (df["gender"] == "M").astype(float)
    elif "gender_M" not in df.columns:
        df["gender_M"] = 0.0

    df = df.sort_values(["stay_id", "day_num"])
    for c in tv_cols:
        df[c] = df.groupby("stay_id")[c].transform(lambda s: s.ffill().fillna(0.0))

    mp = df["mp_j_min"].to_numpy(dtype=np.float64)
    log_mp = np.log(np.clip(mp, 1e-3, None))
    edges_mp, mu_log, sd_log = _build_mp_bin_edges(log_mp, cfg.n_bins, cfg.bin_sd_range)
    df["mp_bin"] = _assign_bins(mp, edges_mp)

    if tv_cols:
        tv_means = df[tv_cols].mean()
        tv_stds = df[tv_cols].std().replace(0, 1.0)
        df[tv_cols] = (df[tv_cols] - tv_means) / tv_stds

    static_per_stay = (
        df.sort_values("day_num").groupby("stay_id")[static_cols].first().fillna(0.0)
    )
    static_per_stay = (
        (static_per_stay - static_per_stay.mean())
        / static_per_stay.std().replace(0, 1.0)
    )

    stays = df["stay_id"].drop_duplicates().to_numpy()
    subject_ids = (
        df[["stay_id", "subject_id"]].drop_duplicates()
        .set_index("stay_id")["subject_id"].reindex(stays).to_numpy()
    )
    N, T, K = len(stays), cfg.max_t, cfg.n_bins
    p_dyn, p_stat = len(tv_cols), len(static_cols)

    stay_to_idx = {sid: i for i, sid in enumerate(stays)}
    df_in = df[(df["day_num"] >= 0) & (df["day_num"] < T)].copy()
    row_i = df_in["stay_id"].map(stay_to_idx).to_numpy()
    row_t = df_in["day_num"].to_numpy().astype(int)
    bin_k = df_in["mp_bin"].to_numpy().astype(int)

    A_bin = np.zeros((N, T, K), dtype=np.float32)
    A_bin[row_i, row_t, bin_k] = 1.0
    L_dyn = np.zeros((N, T, p_dyn), dtype=np.float32)
    if p_dyn > 0:
        L_dyn[row_i, row_t, :] = df_in[tv_cols].to_numpy().astype(np.float32)
    C_static = static_per_stay.reindex(stays).to_numpy().astype(np.float32)

    Y, at_risk = _build_at_risk_and_outcome(df, T)
    Y = Y[:, :, None]
    at_risk = at_risk[:, :, None]

    t_norm = (np.arange(T, dtype=np.float32) / max(T - 1, 1))[None, :, None]
    t_norm = np.broadcast_to(t_norm, (N, T, 1)).astype(np.float32)

    C_broadcast = np.broadcast_to(C_static[:, None, :], (N, T, p_stat)).astype(np.float32)
    drivers = np.concatenate([A_bin, L_dyn, C_broadcast], axis=-1)
    covariates = np.concatenate([drivers, t_norm], axis=-1)

    feature_layout = {
        "n_bins": K,
        "n_dyn": p_dyn,
        "n_static": p_stat,
        "drivers_width": K + p_dyn + p_stat,
        "covariates_width": K + p_dyn + p_stat + 1,
        "tv_cols": tv_cols,
        "static_cols": static_cols,
        "mp_bin_edges": edges_mp.tolist(),
        "log_mp_mu": mu_log,
        "log_mp_sd": sd_log,
        "bin_slice": (0, K),
    }

    return ARDSCohort(
        Y=torch.from_numpy(Y),
        A_bin=torch.from_numpy(A_bin),
        L_dyn=torch.from_numpy(L_dyn),
        C_static=torch.from_numpy(C_static),
        at_risk=torch.from_numpy(at_risk),
        t_norm=torch.from_numpy(t_norm),
        drivers=torch.from_numpy(drivers),
        covariates=torch.from_numpy(covariates),
        bin_edges_mp=edges_mp,
        stay_ids=stays,
        subject_ids=subject_ids,
        severity_label=sev_map.reindex(stays).to_numpy(),
        feature_layout=feature_layout,
    )
