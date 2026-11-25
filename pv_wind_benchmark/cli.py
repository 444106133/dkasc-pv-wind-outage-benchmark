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
    decompose_positive_bias,
    paired_wilcoxon_nmae_test,
    summarize_interval_partition,
    synchronized_block_bootstrap,
)
from .models import (
    PhysicalThermalChainEstimator,
    WindFreeBoostingEstimator,
)


def _evaluate_cohort(
    split_p,
    preds: dict[str, np.ndarray],
    scales,
    bootstrap_cfg,
    reference_name: str = "HGB_WF",
) -> dict:
    metrics_p = {k: compute_point_metrics(split_p, v, scales) for k, v in preds.items()}
    for k, v in preds.items():
        metrics_p[k]["wilcoxon_p_vs_hgb_wf"] = paired_wilcoxon_nmae_test(
            split_p,
            v,
            preds[reference_name],
            scales,
            use_nameplate=True,
        )
    boot_p = synchronized_block_bootstrap(split_p, preds, reference_name, bootstrap_cfg, scales=scales)
    return {
        "n_rows": split_p.n_rows,
        "n_streams": int(len(np.unique(split_p.stream_ids))),
        "point_metrics": metrics_p,
        "block_bootstrap": boot_p,
    }


def _build_fault_masks(split_p, fault_windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    any_fault = np.zeros(split_p.n_rows, dtype=bool)
    major_fault = np.zeros(split_p.n_rows, dtype=bool)
    for fw in fault_windows:
        s_id = int(fw["stream_id"])
        t0 = int(np.datetime64(str(fw["start_date"]) + "T00:00:00", "ns").astype("int64"))
        t1 = int(np.datetime64(str(fw["end_date"]) + "T23:59:59", "ns").astype("int64"))
        m = (split_p.stream_ids == s_id) & (split_p.timestamps_ns >= t0) & (split_p.timestamps_ns <= t1)
        any_fault |= m
        if bool(fw.get("major", True)):
            major_fault |= m
    return any_fault, major_fault


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
    fault_windows = meta_cfg.get("fault_windows", [])
    nameplate_kwp: dict[int, float] = {
        int(k): float(v["nameplate_kwp"])
        for k, v in streams_meta.items()
        if "nameplate_kwp" in v
    }
    fixed_stream_ids = {
        int(k)
        for k, v in streams_meta.items()
        if str(v.get("mounting_type", "Fixed")).lower().startswith("fixed")
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
    t_sub_end = int(np.datetime64(cfg.splits.subtrain_end, "ns").astype("int64"))
    t_train_end = int(np.datetime64(cfg.splits.train_end, "ns").astype("int64"))
    t_trans_end = int(np.datetime64(cfg.splits.transition_end, "ns").astype("int64"))
    t_outage = int(np.datetime64(cfg.splits.wind_outage_timestamp, "ns").astype("int64"))

    train_mask = full_split.timestamps_ns <= t_train_end
    pre_outage_mask = (full_split.timestamps_ns > t_train_end) & (full_split.timestamps_ns <= t_outage) & full_split.wind_valid
    wind_free_mask = full_split.timestamps_ns > t_trans_end
    post_outage_mask = full_split.timestamps_ns > t_outage

    train_split = subset_split(full_split, train_mask)
    eval_pre_outage = subset_split(full_split, pre_outage_mask)
    eval_wind_free = subset_split(full_split, wind_free_mask)
    eval_post_outage = subset_split(full_split, post_outage_mask)

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

    model_phys = PhysicalThermalChainEstimator(cfg.thermal_chain, cfg.thermal_chain.wind_imputer)
    model_phys.fit_wind_imputer(train_split)
    model_phys.fit_power_chain(train_split, scales)

    has_reanalysis = train_split.reanalysis_wind is not None and np.any(np.isfinite(train_split.reanalysis_wind))
    if has_reanalysis:
        model_phys.fit_reanalysis_affine_calibration(train_split)

    poa_cfg = ThermalChainConfig(
        faiman_u0=cfg.thermal_chain.faiman_u0,
        faiman_u1=cfg.thermal_chain.faiman_u1,
        reference_temp_c=cfg.thermal_chain.reference_temp_c,
        reference_irradiance_wm2=cfg.thermal_chain.reference_irradiance_wm2,
        default_gamma_per_c=cfg.thermal_chain.default_gamma_per_c,
        technology_gamma_per_c=cfg.thermal_chain.technology_gamma_per_c,
        technology_trend_rates_per_year=cfg.thermal_chain.technology_trend_rates_per_year,
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
    model_phys_poa.reanalysis_intercept = model_phys.reanalysis_intercept
    model_phys_poa.reanalysis_slope = model_phys.reanalysis_slope
    model_phys_poa.fit_power_chain(train_split, scales)

    model_hgb_wf = WindFreeBoostingEstimator(cfg.hgb_tree).fit(train_split, scales)
    w_imp_train = model_phys.impute_wind(train_split)
    model_hgb_imputed = WindFreeBoostingEstimator(cfg.hgb_tree).fit(train_split, scales, extra_wind=w_imp_train)

    train_valid_w = subset_split(train_split, train_split.wind_valid)
    model_hgb_wind = WindFreeBoostingEstimator(cfg.hgb_tree).fit(
        train_valid_w,
        scales,
        extra_wind=train_valid_w.wind,
    )

    model_hgb_era5 = None
    if has_reanalysis and train_split.reanalysis_wind is not None:
        w_era_train = model_phys.calibrate_reanalysis_wind(train_split.reanalysis_wind)
        model_hgb_era5 = WindFreeBoostingEstimator(cfg.hgb_tree).fit(train_split, scales, extra_wind=w_era_train)

    center_sub = WindFreeBoostingEstimator(cfg.hgb_tree).fit(
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

    results: dict = {"periods": {}}
    for period_name, split_p, has_measured_w in (
        ("pre_outage_2016", eval_pre_outage, True),
        ("wind_free_2017_2026", eval_wind_free, False),
        ("post_outage_2016_2026", eval_post_outage, False),
    ):
        if split_p.n_rows == 0:
            continue
        w_imp = model_phys.impute_wind(split_p)
        preds: dict[str, np.ndarray] = {
            "Phys_Faiman_Imputed": model_phys.predict(split_p, scales, wind_override=w_imp),
            "Phys_POA_Imputed": model_phys_poa.predict(split_p, scales, wind_override=w_imp),
            "HGB_Imputed": model_hgb_imputed.predict(split_p, scales, extra_wind=w_imp),
            "HGB_WF": model_hgb_wf.predict(split_p, scales),
        }
        if has_measured_w:
            preds["Phys_Faiman_MeasuredW"] = model_phys.predict(split_p, scales, wind_override=split_p.wind)
            preds["Phys_POA_MeasuredW"] = model_phys_poa.predict(split_p, scales, wind_override=split_p.wind)
            preds["HGB_Wind"] = model_hgb_wind.predict(split_p, scales, extra_wind=split_p.wind)
        if has_reanalysis and split_p.reanalysis_wind is not None and model_hgb_era5 is not None:
            w_era = model_phys.calibrate_reanalysis_wind(split_p.reanalysis_wind)
            preds["Phys_Faiman_ERA5"] = model_phys.predict(split_p, scales, wind_override=w_era)
            preds["Phys_POA_ERA5"] = model_phys_poa.predict(split_p, scales, wind_override=w_era)
            preds["HGB_ERA5"] = model_hgb_era5.predict(split_p, scales, extra_wind=w_era)

        period_entry: dict = {
            "all_streams": _evaluate_cohort(split_p, preds, scales, cfg.bootstrap),
        }
        if fixed_stream_ids:
            fix_mask = np.isin(split_p.stream_ids, list(fixed_stream_ids))
            if np.any(fix_mask):
                fix_split = subset_split(split_p, fix_mask)
                fix_preds = {k: v[fix_mask] for k, v in preds.items()}
                period_entry["fixed_tilt"] = _evaluate_cohort(fix_split, fix_preds, scales, cfg.bootstrap)

        if period_name == "wind_free_2017_2026":
            monitor = OnlineConformalMonitor(calibrator, cfg.conformal)
            center_p = center_sub.predict(split_p, scales, use_subtrain_scales=True)
            online_res = monitor.evaluate_online_methods(
                split_p,
                center_preds=center_p,
                alarm_reference_preds=preds["HGB_WF"],
                scales=scales,
                level=0.90,
            )
            alarms = monitor.compute_causal_yield_alarms(split_p, preds["HGB_WF"], scales)
            alarm_mask = np.array(
                [alarms.get((int(s), int(d)), (False, 1.0))[0] for s, d in zip(split_p.stream_ids, split_p.day_index)],
                dtype=bool,
            )
            fix_arr = (
                np.isin(split_p.stream_ids, list(fixed_stream_ids))
                if fixed_stream_ids
                else np.ones(split_p.n_rows, dtype=bool)
            )
            any_fault_mask, major_fault_mask = _build_fault_masks(split_p, fault_windows)
            bias_fault_mask = any_fault_mask if np.any(any_fault_mask) else alarm_mask
            period_entry["post_2015_bias_decomposition"] = decompose_positive_bias(
                split_p,
                preds["HGB_WF"],
                scales,
                fixed_tilt_mask=fix_arr,
                fault_mask=bias_fault_mask,
            )
            if np.any(major_fault_mask):
                clean_fixed_mask = fix_arr & (~any_fault_mask)
                period_entry["conformal_om_log_partition"] = {
                    m_name: {
                        "clean_fixed_arrays": summarize_interval_partition(
                            m_val["covered"],
                            m_val["width_normalized"],
                            clean_fixed_mask,
                            m_val["width_kwp_pct"],
                        ),
                        "outside_major_fault_windows": summarize_interval_partition(
                            m_val["covered"],
                            m_val["width_normalized"],
                            ~major_fault_mask,
                            m_val["width_kwp_pct"],
                        ),
                        "post_outage_major_faults": summarize_interval_partition(
                            m_val["covered"],
                            m_val["width_normalized"],
                            major_fault_mask,
                            m_val["width_kwp_pct"],
                        ),
                    }
                    for m_name, m_val in online_res.items()
                }
            period_entry["conformal_yield_alarm_partition"] = {
                m_name: {
                    "alarm_free": summarize_interval_partition(
                        m_val["covered"],
                        m_val["width_normalized"],
                        ~alarm_mask,
                        m_val["width_kwp_pct"],
                    ),
                    "alarmed": summarize_interval_partition(
                        m_val["covered"],
                        m_val["width_normalized"],
                        alarm_mask,
                        m_val["width_kwp_pct"],
                    ),
                    "all": summarize_interval_partition(
                        m_val["covered"],
                        m_val["width_normalized"],
                        None,
                        m_val["width_kwp_pct"],
                    ),
                }
                for m_name, m_val in online_res.items()
            }
            sens_grid: list[dict] = []
            for w_d in (14, 30, 60):
                for tau in (0.70, 0.75, 0.80, 0.85, 0.90):
                    grid_res = monitor.evaluate_online_methods(
                        split_p,
                        center_preds=center_p,
                        alarm_reference_preds=preds["HGB_WF"],
                        scales=scales,
                        level=0.90,
                        yield_threshold=tau,
                        rolling_window_days=w_d,
                    )
                    fg_asym = grid_res["FaultGated_Asym"]
                    sens_grid.append(
                        {
                            "rolling_window_days": w_d,
                            "yield_threshold": tau,
                            "alarm_free": summarize_interval_partition(
                                fg_asym["covered"],
                                fg_asym["width_normalized"],
                                ~alarm_mask,
                                fg_asym["width_kwp_pct"],
                            ),
                            "alarmed": summarize_interval_partition(
                                fg_asym["covered"],
                                fg_asym["width_normalized"],
                                alarm_mask,
                                fg_asym["width_kwp_pct"],
                            ),
                        }
                    )
            period_entry["fault_gate_sensitivity"] = sens_grid

        results["periods"][period_name] = period_entry

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
