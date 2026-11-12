from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import BenchmarkConfig, ThermalChainConfig
from .conformal import OnlineConformalMonitor, SplitConformalCalibrator
from .data import (
    apply_service_windows,
    compute_fleet_scales,
    load_json_config,
    prepare_dataframe_to_split,
    subset_split,
)
from .evaluation import (
    compute_point_metrics,
    paired_wilcoxon_nmae_test,
    summarize_interval_partition,
    synchronized_block_bootstrap,
)
from .models import (
    AvailabilityGatedMixtureOfExperts,
    MeasuredWindFallbackHybrid,
    PhysicalThermalChainEstimator,
    WindFreeBoostingEstimator,
    WindValueDistillationGate,
)


def run_benchmark_pipeline(
    data_path: str | Path,
    config_path: str | Path,
    tech_codes_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
) -> dict:
    cfg = BenchmarkConfig.from_json(config_path)
    tech_cfg = load_json_config(tech_codes_path)
    meta_cfg = load_json_config(metadata_path)

    tech_code_map: dict[str, int] = {str(k): int(v) for k, v in tech_cfg["code_of_technology"].items()}
    tech_names: dict[int, str] = {int(v): str(k) for k, v in tech_code_map.items()}
    streams_meta = meta_cfg.get("streams", {})
    nameplate_kwp: dict[int, float] = {
        int(k): float(v["nameplate_kwp"])
        for k, v in streams_meta.items()
        if "nameplate_kwp" in v
    }
    exclusions: list[dict] = []
    for k, v in streams_meta.items():
        if v.get("service_start"):
            exclusions.append({"system_id": int(k), "exclude_before": v["service_start"]})
        if v.get("service_end"):
            exclusions.append({"system_id": int(k), "exclude_after": v["service_end"]})

    p = Path(data_path)
    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    df = apply_service_windows(
        df,
        cfg.schema.timestamp_col,
        cfg.schema.stream_id_col,
        exclusions,
    )

    full_split = prepare_dataframe_to_split(df, cfg, tech_code_map)
    t_sub_end = pd.Timestamp(cfg.splits.subtrain_end).value
    t_cal_split = pd.Timestamp(cfg.splits.calibration_split_timestamp).value
    t_train_end = pd.Timestamp(cfg.splits.train_end).value
    t_trans_end = pd.Timestamp(cfg.splits.transition_end).value
    t_outage = pd.Timestamp(cfg.splits.wind_outage_timestamp).value

    train_mask = full_split.timestamps_ns <= t_train_end
    eval_transition_mask = (full_split.timestamps_ns > t_train_end) & (full_split.timestamps_ns <= t_trans_end)
    eval_post_outage_mask = full_split.timestamps_ns > t_trans_end

    train_split = subset_split(full_split, train_mask)
    eval_transition = subset_split(full_split, eval_transition_mask)
    eval_post_outage = subset_split(full_split, eval_post_outage_mask)

    subtrain_mask = train_split.timestamps_ns <= t_sub_end
    fwd_mask = train_split.timestamps_ns > t_sub_end
    doy_frac = np.mod(np.mod(np.arctan2(train_split.doy_sin, train_split.doy_cos) / (2.0 * np.pi), 1.0) + 1.0, 1.0)
    early_cal_mask = fwd_mask & (doy_frac < 0.5)
    late_cal_mask = fwd_mask & (~(doy_frac < 0.5))

    scales = compute_fleet_scales(
        train_split,
        subtrain_mask=subtrain_mask,
        technology_names=tech_names,
        nameplate_kwp=nameplate_kwp,
        quantile=cfg.splits.scale_quantile,
        min_rows=cfg.splits.min_train_rows_for_stream_scale,
    )

    model_wf_base = WindFreeBoostingEstimator(cfg.baseline_tree).fit(train_split, scales)
    model_fb_hybrid = MeasuredWindFallbackHybrid(cfg.fallback_hybrid).fit(train_split, scales)
    model_gated_moe = AvailabilityGatedMixtureOfExperts(cfg.mixture_of_experts, cfg.baseline_tree).fit(
        train_split,
        subtrain_mask=subtrain_mask,
        forward_2015_mask=fwd_mask,
        scales=scales,
    )
    model_phys = PhysicalThermalChainEstimator(cfg.thermal_chain, cfg.thermal_chain.wind_imputer)
    model_phys.fit_wind_imputer(train_split)
    model_phys.fit_power_chain(train_split, scales)
    model_tuned_wf = WindFreeBoostingEstimator(cfg.tuned_tree).fit(train_split, scales)

    poa_cfg = ThermalChainConfig(
        faiman_u0=cfg.thermal_chain.faiman_u0,
        faiman_u1=cfg.thermal_chain.faiman_u1,
        reference_temp_c=cfg.thermal_chain.reference_temp_c,
        reference_irradiance_wm2=cfg.thermal_chain.reference_irradiance_wm2,
        default_gamma_per_c=cfg.thermal_chain.default_gamma_per_c,
        technology_gamma_per_c=cfg.thermal_chain.technology_gamma_per_c,
        technology_degradation_per_year=cfg.thermal_chain.technology_degradation_per_year,
        use_datasheet_gamma=True,
        use_harmonic_poa_transposition=True,
        clip_predictions=True,
        max_normalized_power=cfg.thermal_chain.max_normalized_power,
        transposition_ridge_lambda=cfg.thermal_chain.transposition_ridge_lambda,
        response_ridge_scale=cfg.thermal_chain.response_ridge_scale,
        wind_imputer=cfg.thermal_chain.wind_imputer,
    )
    model_phys_poa = PhysicalThermalChainEstimator(poa_cfg, cfg.thermal_chain.wind_imputer)
    model_phys_poa.wind_imputer = model_phys.wind_imputer
    model_phys_poa.fit_power_chain(train_split, scales)

    gate_eval = WindValueDistillationGate(cfg.tuned_tree, cfg.inference).fit_and_evaluate_gate(
        train_split,
        subtrain_mask=subtrain_mask,
        forward_2015_mask=fwd_mask,
        scales=scales,
    )

    center_sub = WindFreeBoostingEstimator(cfg.tuned_tree).fit(
        subset_split(train_split, subtrain_mask),
        scales,
        use_subtrain_scales=True,
    )
    early_cal = subset_split(train_split, early_cal_mask)
    late_cal = subset_split(train_split, late_cal_mask)
    calibrator = SplitConformalCalibrator(cfg.conformal, cfg.conformal.scale_tree).fit(
        early_cal,
        center_sub.predict(early_cal, scales, use_subtrain_scales=True),
        late_cal,
        center_sub.predict(late_cal, scales, use_subtrain_scales=True),
        scales=scales,
    )

    results: dict = {"teacher_gate": gate_eval, "periods": {}}
    for period_name, split_p, allow_w in (
        ("transition_period", eval_transition, True),
        ("post_outage_period", eval_post_outage, False),
    ):
        if split_p.n_rows == 0:
            continue
        preds = {
            "WindFree_Baseline": model_wf_base.predict(split_p, scales),
            "Fallback_Hybrid": model_fb_hybrid.predict(split_p, scales, allow_wind=allow_w),
            "Gated_MoE": model_gated_moe.predict(split_p, scales, allow_wind=allow_w),
            "Physical_Impute": model_phys.predict(split_p, scales),
            "Physical_POA_Transposition": model_phys_poa.predict(split_p, scales),
            "Tuned_WindFree": model_tuned_wf.predict(split_p, scales),
        }
        metrics_p = {k: compute_point_metrics(split_p, v, scales) for k, v in preds.items()}
        for k, v in preds.items():
            metrics_p[k]["wilcoxon_p_vs_tuned_wf"] = paired_wilcoxon_nmae_test(
                split_p,
                v,
                preds["Tuned_WindFree"],
                scales,
            )
        boot_p = synchronized_block_bootstrap(split_p, preds, "Tuned_WindFree", cfg.inference)
        results["periods"][period_name] = {
            "n_rows": split_p.n_rows,
            "point_metrics": metrics_p,
            "block_bootstrap": boot_p,
        }

        if period_name == "transition_period":
            pre_m = split_p.timestamps_ns <= t_outage
            post_m = split_p.timestamps_ns > t_outage
            if np.any(pre_m) and np.any(post_m):
                pre_s = subset_split(split_p, pre_m)
                post_s = subset_split(split_p, post_m)
                results["periods"]["transition_pre_outage"] = {
                    k: compute_point_metrics(pre_s, v[pre_m], scales) for k, v in preds.items()
                }
                results["periods"]["transition_post_outage"] = {
                    k: compute_point_metrics(post_s, v[post_m], scales) for k, v in preds.items()
                }

        if period_name == "post_outage_period":
            monitor = OnlineConformalMonitor(calibrator, cfg.conformal)
            center_p = center_sub.predict(split_p, scales, use_subtrain_scales=True)
            online_res = monitor.evaluate_online_methods(
                split_p,
                center_preds=center_p,
                alarm_reference_preds=preds["Tuned_WindFree"],
                scales=scales,
                level=0.90,
            )
            alarms = monitor.compute_causal_yield_alarms(split_p, preds["Tuned_WindFree"], scales)
            alarm_mask = np.array(
                [alarms.get((int(s), int(d)), (False, 1.0))[0] for s, d in zip(split_p.stream_ids, split_p.day_index)],
                dtype=bool,
            )
            results["periods"][period_name]["conformal_yield_alarm_partition"] = {
                m_name: {
                    "alarm_free": summarize_interval_partition(
                        m_val["covered"],
                        m_val["width_normalized"],
                        ~alarm_mask,
                    ),
                    "alarmed": summarize_interval_partition(
                        m_val["covered"],
                        m_val["width_normalized"],
                        alarm_mask,
                    ),
                    "all": summarize_interval_partition(
                        m_val["covered"],
                        m_val["width_normalized"],
                        None,
                    ),
                }
                for m_name, m_val in online_res.items()
            }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_file = out_path / "benchmark_summary.json"
    summary_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/benchmark_config.json")
    parser.add_argument("--tech-codes", default="configs/technology_codes.json")
    parser.add_argument("--metadata", default="configs/site_metadata.json")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_benchmark_pipeline(
        data_path=args.data,
        config_path=args.config,
        tech_codes_path=args.tech_codes,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
