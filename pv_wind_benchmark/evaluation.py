from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from .config import BootstrapConfig
from .data import FleetScales, PreparedSplit


def compute_point_metrics(
    split: PreparedSplit,
    predictions: np.ndarray,
    scales: FleetScales,
) -> dict[str, float]:
    err = predictions - split.power
    abs_err = np.abs(err)
    micro_mae = float(np.mean(abs_err))
    micro_rmse = float(np.sqrt(np.mean(err * err)))
    micro_mbe = float(np.mean(err))

    kwp_arr = np.array(
        [scales.nameplate_kwp.get(int(s), scales.stream_scales[int(s)]) for s in split.stream_ids],
        dtype=np.float64,
    )
    err_kwp_pct = 100.0 * err / kwp_arr
    y_kwp_pct = 100.0 * split.power / kwp_arr
    ss_res_pool = float(np.sum(err_kwp_pct * err_kwp_pct))
    ss_tot_pool = float(np.sum((y_kwp_pct - float(np.mean(y_kwp_pct))) ** 2))
    pooled_r2_kwp = 1.0 - ss_res_pool / ss_tot_pool if ss_tot_pool > 0.0 else 0.0

    streams = sorted(int(s) for s in np.unique(split.stream_ids))
    nmae_cs_list: list[float] = []
    nmae_kwp_list: list[float] = []
    nrmse_kwp_list: list[float] = []
    nmbe_cs_list: list[float] = []
    nmbe_kwp_list: list[float] = []
    r2_list: list[float] = []

    for s in streams:
        ms = split.stream_ids == s
        c_s = scales.stream_scales[s]
        kwp_s = scales.nameplate_kwp.get(s, c_s)
        e_s = err[ms]
        y_s = split.power[ms]
        mae_s = float(np.mean(np.abs(e_s)))
        rmse_s = float(np.sqrt(np.mean(e_s * e_s)))
        mbe_s = float(np.mean(e_s))
        nmae_cs_list.append(100.0 * mae_s / c_s)
        nmae_kwp_list.append(100.0 * mae_s / kwp_s)
        nrmse_kwp_list.append(100.0 * rmse_s / kwp_s)
        nmbe_cs_list.append(100.0 * mbe_s / c_s)
        nmbe_kwp_list.append(100.0 * mbe_s / kwp_s)
        ss_tot_s = float(np.sum((y_s - float(np.mean(y_s))) ** 2))
        if ss_tot_s > 0.0:
            r2_list.append(1.0 - float(np.sum(e_s * e_s)) / ss_tot_s)

    return {
        "micro_mae": micro_mae,
        "micro_rmse": micro_rmse,
        "micro_mbe": micro_mbe,
        "pooled_nmae_kwp_pct": float(np.mean(np.abs(err_kwp_pct))),
        "pooled_nrmse_kwp_pct": float(np.sqrt(np.mean(err_kwp_pct * err_kwp_pct))),
        "pooled_nmbe_kwp_pct": float(np.mean(err_kwp_pct)),
        "pooled_r2_kwp": pooled_r2_kwp,
        "median_nmae_cs_pct": float(np.median(nmae_cs_list)),
        "median_nmae_kwp_pct": float(np.median(nmae_kwp_list)),
        "median_nrmse_kwp_pct": float(np.median(nrmse_kwp_list)),
        "median_nmbe_cs_pct": float(np.median(nmbe_cs_list)),
        "median_nmbe_kwp_pct": float(np.median(nmbe_kwp_list)),
        "median_r2": float(np.median(r2_list)) if r2_list else 0.0,
    }


def paired_wilcoxon_nmae_test(
    split: PreparedSplit,
    preds_candidate: np.ndarray,
    preds_reference: np.ndarray,
    scales: FleetScales,
    use_nameplate: bool = True,
) -> float:
    streams = sorted(int(s) for s in np.unique(split.stream_ids))
    diffs: list[float] = []
    for s in streams:
        ms = split.stream_ids == s
        denom = scales.nameplate_kwp.get(s, scales.stream_scales[s]) if use_nameplate else scales.stream_scales[s]
        mae_cand = 100.0 * float(np.mean(np.abs(preds_candidate[ms] - split.power[ms]))) / denom
        mae_ref = 100.0 * float(np.mean(np.abs(preds_reference[ms] - split.power[ms]))) / denom
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
    config: BootstrapConfig,
    scales: FleetScales | None = None,
) -> dict[str, dict[str, float | list[float]]]:
    unique_days = np.sort(np.unique(split.day_index))
    block_len = config.block_length_days
    n_blocks = max(1, int(np.ceil(len(unique_days) / block_len)))
    day_to_block = {int(d): int(i // block_len) for i, d in enumerate(unique_days)}
    row_blocks = np.array([day_to_block[int(d)] for d in split.day_index], dtype=np.int32)

    kwp_arr = (
        np.array(
            [scales.nameplate_kwp.get(int(s), scales.stream_scales[int(s)]) for s in split.stream_ids],
            dtype=np.float64,
        )
        if scales is not None
        else np.ones(split.n_rows, dtype=np.float64)
    )

    block_counts = np.bincount(row_blocks, minlength=n_blocks).astype(np.float64)
    block_err_sums: dict[str, np.ndarray] = {}
    block_nmae_sums: dict[str, np.ndarray] = {}
    point_mae: dict[str, float] = {}
    point_nmae: dict[str, float] = {}

    for name, preds in model_predictions.items():
        abs_e = np.abs(preds - split.power)
        abs_ne = 100.0 * abs_e / kwp_arr
        point_mae[name] = float(np.mean(abs_e))
        point_nmae[name] = float(np.mean(abs_ne))
        sums = np.zeros(n_blocks, dtype=np.float64)
        nsums = np.zeros(n_blocks, dtype=np.float64)
        np.add.at(sums, row_blocks, abs_e)
        np.add.at(nsums, row_blocks, abs_ne)
        block_err_sums[name] = sums
        block_nmae_sums[name] = nsums

    rng = np.random.default_rng(config.random_seed)
    boot_mae: dict[str, np.ndarray] = {
        name: np.zeros(config.n_bootstrap_draws, dtype=np.float64) for name in model_predictions
    }
    boot_nmae: dict[str, np.ndarray] = {
        name: np.zeros(config.n_bootstrap_draws, dtype=np.float64) for name in model_predictions
    }

    for b in range(config.n_bootstrap_draws):
        b_idx = rng.integers(0, n_blocks, size=n_blocks)
        denom = float(np.sum(block_counts[b_idx]))
        for name in model_predictions:
            boot_mae[name][b] = float(np.sum(block_err_sums[name][b_idx])) / denom
            boot_nmae[name][b] = float(np.sum(block_nmae_sums[name][b_idx])) / denom

    results: dict[str, dict[str, float | list[float]]] = {}
    ref_draws = boot_mae[reference_name]
    ref_ndraws = boot_nmae[reference_name]
    ref_point = point_mae[reference_name]
    ref_npoint = point_nmae[reference_name]

    for name in model_predictions:
        draws = boot_mae[name]
        diff_draws = draws - ref_draws
        p_pos = float(np.mean(diff_draws >= 0.0))
        p_neg = float(np.mean(diff_draws <= 0.0))
        p_two = min(1.0, 2.0 * min(p_pos, p_neg))

        ndraws = boot_nmae[name]
        ndiff_draws = ndraws - ref_ndraws
        np_pos = float(np.mean(ndiff_draws >= 0.0))
        np_neg = float(np.mean(ndiff_draws <= 0.0))
        np_two = min(1.0, 2.0 * min(np_pos, np_neg))

        results[name] = {
            "micro_mae": point_mae[name],
            "mae_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "delta_mae_vs_ref": point_mae[name] - ref_point,
            "delta_mae_ci95": [float(np.quantile(diff_draws, 0.025)), float(np.quantile(diff_draws, 0.975))],
            "p_bootstrap_two_sided": p_two,
            "pooled_nmae_kwp_pct": point_nmae[name],
            "nmae_kwp_ci95": [float(np.quantile(ndraws, 0.025)), float(np.quantile(ndraws, 0.975))],
            "delta_nmae_kwp_vs_ref": point_nmae[name] - ref_npoint,
            "delta_nmae_kwp_ci95": [float(np.quantile(ndiff_draws, 0.025)), float(np.quantile(ndiff_draws, 0.975))],
            "p_bootstrap_nmae_kwp": np_two,
        }
    return results


def decompose_positive_bias(
    split: PreparedSplit,
    predictions: np.ndarray,
    scales: FleetScales,
    fixed_tilt_mask: np.ndarray,
    fault_mask: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    kwp_arr = np.array(
        [scales.nameplate_kwp.get(int(s), scales.stream_scales[int(s)]) for s in split.stream_ids],
        dtype=np.float64,
    )
    err_kwp = 100.0 * (predictions - split.power) / kwp_arr
    total_rows = max(1, split.n_rows)
    total_bias_sum = float(np.sum(err_kwp))

    groups = {
        "logged_faults": fault_mask,
        "trackers_cpv_outside_faults": (~fault_mask) & (~fixed_tilt_mask),
        "clean_fixed_tilt": (~fault_mask) & fixed_tilt_mask,
    }
    out: dict[str, dict[str, float | int]] = {}
    for g_name, g_mask in groups.items():
        n_g = int(np.sum(g_mask))
        sum_g = float(np.sum(err_kwp[g_mask])) if n_g > 0 else 0.0
        out[g_name] = {
            "n_rows": n_g,
            "row_share_pct": 100.0 * n_g / total_rows,
            "mean_nmbe_kwp_pct": float(np.mean(err_kwp[g_mask])) if n_g > 0 else 0.0,
            "bias_share_pct": (100.0 * sum_g / total_bias_sum) if abs(total_bias_sum) > 1e-12 else 0.0,
        }
    return out


def summarize_interval_partition(
    covered: np.ndarray,
    width_normalized: np.ndarray,
    mask: np.ndarray | None = None,
    width_kwp_pct: np.ndarray | None = None,
) -> dict[str, float | int]:
    if mask is not None:
        cov = covered[mask]
        wid = width_normalized[mask]
        w_kwp = width_kwp_pct[mask] if width_kwp_pct is not None else None
    else:
        cov = covered
        wid = width_normalized
        w_kwp = width_kwp_pct
    res: dict[str, float | int] = {
        "n_rows": int(len(cov)),
        "micro_coverage_pct": float(100.0 * np.mean(cov)) if len(cov) > 0 else 0.0,
        "mean_normalized_width": float(np.mean(wid)) if len(wid) > 0 else 0.0,
    }
    if w_kwp is not None:
        res["mean_width_kwp_pct"] = float(np.mean(w_kwp)) if len(w_kwp) > 0 else 0.0
    return res
