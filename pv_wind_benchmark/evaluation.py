from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from .config import GateAndBootstrapConfig
from .data import FleetScales, PreparedSplit


def compute_point_metrics(
    split: PreparedSplit,
    predictions: np.ndarray,
    scales: FleetScales,
) -> dict[str, float]:
    err = predictions - split.power
    abs_err = np.abs(err)
    micro_mae = float(np.mean(abs_err))

    streams = sorted(int(s) for s in np.unique(split.stream_ids))
    nmae_cs_list: list[float] = []
    nmae_kwp_list: list[float] = []
    nmbe_cs_list: list[float] = []

    for s in streams:
        ms = split.stream_ids == s
        c_s = scales.stream_scales[s]
        kwp_s = scales.nameplate_kwp.get(s, c_s)
        mae_s = float(np.mean(abs_err[ms]))
        mbe_s = float(np.mean(err[ms]))
        nmae_cs_list.append(100.0 * mae_s / c_s)
        nmae_kwp_list.append(100.0 * mae_s / kwp_s)
        nmbe_cs_list.append(100.0 * mbe_s / c_s)

    return {
        "micro_mae": micro_mae,
        "median_nmae_cs_pct": float(np.median(nmae_cs_list)),
        "median_nmae_kwp_pct": float(np.median(nmae_kwp_list)),
        "median_nmbe_cs_pct": float(np.median(nmbe_cs_list)),
    }


def paired_wilcoxon_nmae_test(
    split: PreparedSplit,
    preds_candidate: np.ndarray,
    preds_reference: np.ndarray,
    scales: FleetScales,
) -> float:
    streams = sorted(int(s) for s in np.unique(split.stream_ids))
    diffs: list[float] = []
    for s in streams:
        ms = split.stream_ids == s
        c_s = scales.stream_scales[s]
        mae_cand = float(np.mean(np.abs(preds_candidate[ms] - split.power[ms]))) / c_s
        mae_ref = float(np.mean(np.abs(preds_reference[ms] - split.power[ms]))) / c_s
        diffs.append(mae_cand - mae_ref)
    arr = np.asarray(diffs, dtype=np.float64)
    if np.allclose(arr, 0.0):
        return 1.0
    _, p_val = wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
    return float(p_val)


def synchronized_block_bootstrap(
    split: PreparedSplit,
    model_predictions: dict[str, np.ndarray],
    reference_name: str,
    config: GateAndBootstrapConfig,
) -> dict[str, dict[str, float | list[float]]]:
    unique_days = np.sort(np.unique(split.day_index))
    block_len = config.block_length_days
    n_blocks = max(1, int(np.ceil(len(unique_days) / block_len)))
    day_to_block = {int(d): int(i // block_len) for i, d in enumerate(unique_days)}
    row_blocks = np.array([day_to_block[int(d)] for d in split.day_index], dtype=np.int32)

    block_counts = np.bincount(row_blocks, minlength=n_blocks).astype(np.float64)
    block_err_sums: dict[str, np.ndarray] = {}
    point_mae: dict[str, float] = {}

    for name, preds in model_predictions.items():
        abs_e = np.abs(preds - split.power)
        point_mae[name] = float(np.mean(abs_e))
        sums = np.zeros(n_blocks, dtype=np.float64)
        np.add.at(sums, row_blocks, abs_e)
        block_err_sums[name] = sums

    rng = np.random.default_rng(config.random_seed)
    boot_mae: dict[str, np.ndarray] = {
        name: np.zeros(config.n_bootstrap_draws, dtype=np.float64) for name in model_predictions
    }

    for b in range(config.n_bootstrap_draws):
        b_idx = rng.integers(0, n_blocks, size=n_blocks)
        denom = float(np.sum(block_counts[b_idx]))
        for name, sums in block_err_sums.items():
            boot_mae[name][b] = float(np.sum(sums[b_idx])) / denom

    results: dict[str, dict[str, float | list[float]]] = {}
    ref_draws = boot_mae[reference_name]
    ref_point = point_mae[reference_name]

    for name in model_predictions:
        draws = boot_mae[name]
        diff_draws = draws - ref_draws
        p_pos = float(np.mean(diff_draws >= 0.0))
        p_neg = float(np.mean(diff_draws <= 0.0))
        p_two = min(1.0, 2.0 * min(p_pos, p_neg))
        results[name] = {
            "micro_mae": point_mae[name],
            "mae_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "delta_mae_vs_ref": point_mae[name] - ref_point,
            "delta_mae_ci95": [float(np.quantile(diff_draws, 0.025)), float(np.quantile(diff_draws, 0.975))],
            "p_bootstrap_two_sided": p_two,
        }
    return results


def summarize_interval_partition(
    covered: np.ndarray,
    width_normalized: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    if mask is not None:
        cov = covered[mask]
        wid = width_normalized[mask]
    else:
        cov = covered
        wid = width_normalized
    return {
        "n_rows": int(len(cov)),
        "micro_coverage_pct": float(100.0 * np.mean(cov)) if len(cov) > 0 else 0.0,
        "mean_normalized_width": float(np.mean(wid)) if len(wid) > 0 else 0.0,
    }
