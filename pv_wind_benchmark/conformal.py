from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ConformalConfig, TreeConfig
from .data import FleetScales, PreparedSplit, subset_split


def weighted_quantile(values: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return 1.0
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    k = min(n - 1, max(0, int(np.ceil((n + 1) * quantile)) - 1))
    return float(v[k])


class SplitConformalCalibrator:
    def __init__(self, config: ConformalConfig, scale_tree_config: TreeConfig) -> None:
        self.config = config
        self.scale_tree_config = scale_tree_config
        self.scale_model: HistGradientBoostingRegressor | None = None
        self.stream_quantiles: dict[float, dict[int, float]] = {}
        self.pooled_quantiles: dict[float, float] = {}
        self.stream_asym_quantiles: dict[float, dict[int, tuple[float, float]]] = {}
        self.pooled_asym_quantiles: dict[float, tuple[float, float]] = {}
        self.width_thresholds: dict[float, float] = {}

    @staticmethod
    def _scale_features(split: PreparedSplit) -> np.ndarray:
        return split.wind_free_matrix()

    def predict_residual_scale(self, split: PreparedSplit, scales: FleetScales | None = None) -> np.ndarray:
        if self.scale_model is None:
            raise RuntimeError("Residual scale model has not been fitted.")
        x = self._scale_features(split)
        raw = self.scale_model.predict(x)
        if scales is not None:
            c = scales.scale_array(split.stream_ids, subtrain=True)
            return np.maximum(self.config.scale_floor, c * raw) / c
        return np.maximum(self.config.scale_floor, raw)

    def fit(
        self,
        early_cal_split: PreparedSplit,
        early_center_preds: np.ndarray,
        late_cal_split: PreparedSplit,
        late_center_preds: np.ndarray,
        scales: FleetScales,
    ) -> SplitConformalCalibrator:
        c_early = scales.scale_array(early_cal_split.stream_ids, subtrain=True)
        abs_res_early = np.abs(early_cal_split.power - early_center_preds) / c_early

        self.scale_model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.scale_tree_config.learning_rate,
            max_iter=self.scale_tree_config.max_iter,
            max_leaf_nodes=self.scale_tree_config.max_leaf_nodes,
            max_depth=self.scale_tree_config.max_depth,
            min_samples_leaf=self.scale_tree_config.min_samples_leaf,
            l2_regularization=self.scale_tree_config.l2_regularization,
            early_stopping=self.scale_tree_config.early_stopping,
            random_state=self.scale_tree_config.random_seed,
        )
        self.scale_model.fit(self._scale_features(early_cal_split), abs_res_early)

        c_late = scales.scale_array(late_cal_split.stream_ids, subtrain=True)
        sigma_late = self.predict_residual_scale(late_cal_split, scales=scales)
        scores_sym = np.abs(late_cal_split.power - late_center_preds) / (c_late * sigma_late)
        scores_low = np.maximum(0.0, late_center_preds - late_cal_split.power) / (c_late * sigma_late)
        scores_high = np.maximum(0.0, late_cal_split.power - late_center_preds) / (c_late * sigma_late)

        streams = sorted(int(s) for s in np.unique(late_cal_split.stream_ids))
        for level in self.config.nominal_levels:
            alpha = 1.0 - level
            q_pool = weighted_quantile(scores_sym, level)
            q_low_pool = weighted_quantile(scores_low, 1.0 - 0.5 * alpha)
            q_high_pool = weighted_quantile(scores_high, 1.0 - 0.5 * alpha)
            self.pooled_quantiles[level] = q_pool
            self.pooled_asym_quantiles[level] = (q_low_pool, q_high_pool)
            self.stream_quantiles[level] = {}
            self.stream_asym_quantiles[level] = {}

            for s in streams:
                ms = late_cal_split.stream_ids == s
                if int(np.sum(ms)) >= self.config.min_stream_calibration_rows:
                    self.stream_quantiles[level][s] = weighted_quantile(scores_sym[ms], level)
                    self.stream_asym_quantiles[level][s] = (
                        weighted_quantile(scores_low[ms], 1.0 - 0.5 * alpha),
                        weighted_quantile(scores_high[ms], 1.0 - 0.5 * alpha),
                    )
                else:
                    self.stream_quantiles[level][s] = q_pool
                    self.stream_asym_quantiles[level][s] = (q_low_pool, q_high_pool)

            q_arr = np.array([self.stream_quantiles[level].get(int(s), q_pool) for s in late_cal_split.stream_ids])
            lo = np.maximum(0.0, late_center_preds - q_arr * sigma_late * c_late)
            hi = late_center_preds + q_arr * sigma_late * c_late
            norm_width = (hi - lo) / c_late
            self.width_thresholds[level] = float(np.quantile(norm_width, self.config.retention_width_quantile))

        return self

    def static_intervals(
        self,
        split: PreparedSplit,
        center_preds: np.ndarray,
        scales: FleetScales,
        level: float = 0.90,
        asymmetric: bool = False,
    ) -> dict[str, np.ndarray]:
        c = scales.scale_array(split.stream_ids, subtrain=True)
        sigma = self.predict_residual_scale(split, scales=scales)
        if not asymmetric:
            q_pool = self.pooled_quantiles[level]
            q_map = self.stream_quantiles[level]
            q = np.array([q_map.get(int(s), q_pool) for s in split.stream_ids], dtype=np.float64)
            lo = np.maximum(0.0, center_preds - q * sigma * c)
            hi = center_preds + q * sigma * c
        else:
            ql_pool, qh_pool = self.pooled_asym_quantiles[level]
            q_map = self.stream_asym_quantiles[level]
            ql = np.array([q_map.get(int(s), (ql_pool, qh_pool))[0] for s in split.stream_ids], dtype=np.float64)
            qh = np.array([q_map.get(int(s), (ql_pool, qh_pool))[1] for s in split.stream_ids], dtype=np.float64)
            lo = np.maximum(0.0, center_preds - ql * sigma * c)
            hi = center_preds + qh * sigma * c

        width_norm = (hi - lo) / c
        tau = self.width_thresholds[level]
        retained = width_norm <= tau
        covered = (split.power >= lo) & (split.power <= hi)
        return {
            "lower": lo,
            "upper": hi,
            "width_normalized": width_norm,
            "retained": retained,
            "covered": covered,
        }


class OnlineConformalMonitor:
    def __init__(self, calibrator: SplitConformalCalibrator, config: ConformalConfig) -> None:
        self.calibrator = calibrator
        self.config = config

    def compute_causal_yield_alarms(
        self,
        split: PreparedSplit,
        expected_power: np.ndarray,
        scales: FleetScales,
    ) -> dict[tuple[int, int], tuple[bool, float]]:
        c = scales.scale_array(split.stream_ids)
        high_mask = expected_power >= (self.config.fault_gate_min_expected_fraction * c)
        alarms: dict[tuple[int, int], tuple[bool, float]] = {}

        for s in sorted(int(v) for v in np.unique(split.stream_ids)):
            ms = split.stream_ids == s
            days = np.sort(np.unique(split.day_index[ms]))
            day_obs: dict[int, float] = {}
            day_exp: dict[int, float] = {}
            day_cnt: dict[int, int] = {}

            ms_high = ms & high_mask
            for d in days:
                md = ms_high & (split.day_index == d)
                cnt = int(np.sum(md))
                day_cnt[int(d)] = cnt
                if cnt > 0:
                    day_obs[int(d)] = float(np.sum(split.power[md]))
                    day_exp[int(d)] = float(np.sum(expected_power[md]))
                else:
                    day_obs[int(d)] = 0.0
                    day_exp[int(d)] = 0.0

            w_days = self.config.fault_gate_trailing_days
            for d in days:
                di = int(d)
                tot_cnt = sum(day_cnt.get(tau, 0) for tau in range(di - w_days, di))
                tot_exp = sum(day_exp.get(tau, 0.0) for tau in range(di - w_days, di))
                tot_obs = sum(day_obs.get(tau, 0.0) for tau in range(di - w_days, di))
                if tot_cnt >= self.config.fault_gate_min_high_irradiance_points and tot_exp > 0.0:
                    rho = tot_obs / tot_exp
                    is_alarm = bool(rho < self.config.fault_gate_yield_threshold)
                else:
                    rho = 1.0
                    is_alarm = False
                alarms[(s, di)] = (is_alarm, float(rho))
        return alarms

    def evaluate_online_methods(
        self,
        split: PreparedSplit,
        center_preds: np.ndarray,
        alarm_reference_preds: np.ndarray,
        scales: FleetScales,
        level: float = 0.90,
    ) -> dict[str, dict[str, np.ndarray]]:
        c = scales.scale_array(split.stream_ids, subtrain=True)
        sigma = self.calibrator.predict_residual_scale(split, scales=scales)
        scores_sym = np.abs(split.power - center_preds) / (c * sigma)
        scores_low = np.maximum(0.0, center_preds - split.power) / (c * sigma)
        scores_high = np.maximum(0.0, split.power - center_preds) / (c * sigma)

        alarms = self.compute_causal_yield_alarms(split, alarm_reference_preds, scales)
        alpha = 1.0 - level
        w_days = self.config.rolling_window_days
        min_pts = self.config.min_rolling_residuals

        methods = (
            "Static",
            "Static_Asym",
            "Rolling",
            "Causal_ACI",
            "Asym_Rolling",
            "FaultGated_Rolling",
            "FaultGated_Asym",
        )
        out: dict[str, dict[str, np.ndarray]] = {
            m: {
                "covered": np.zeros(split.n_rows, dtype=bool),
                "width_normalized": np.zeros(split.n_rows, dtype=np.float64),
            }
            for m in methods
        }

        static_sym = self.calibrator.static_intervals(split, center_preds, scales, level=level, asymmetric=False)
        static_asym = self.calibrator.static_intervals(split, center_preds, scales, level=level, asymmetric=True)
        out["Static"]["covered"] = static_sym["covered"]
        out["Static"]["width_normalized"] = static_sym["width_normalized"]
        out["Static_Asym"]["covered"] = static_asym["covered"]
        out["Static_Asym"]["width_normalized"] = static_asym["width_normalized"]

        q_pool = self.calibrator.pooled_quantiles[level]
        ql_pool, qh_pool = self.calibrator.pooled_asym_quantiles[level]

        for s in sorted(int(v) for v in np.unique(split.stream_ids)):
            ms = split.stream_ids == s
            q0 = self.calibrator.stream_quantiles[level].get(s, q_pool)
            ql0, qh0 = self.calibrator.stream_asym_quantiles[level].get(s, (ql_pool, qh_pool))

            days = np.sort(np.unique(split.day_index[ms]))
            day_rows: dict[int, np.ndarray] = {}
            for d in days:
                day_rows[int(d)] = np.flatnonzero(ms & (split.day_index == d))

            alpha_t = alpha
            q_fg_last = q0
            ql_fg_last, qh_fg_last = ql0, qh0

            for d in days:
                di = int(d)
                idx = day_rows[di]
                is_alarm, _ = alarms.get((s, di), (False, 1.0))

                hist_idx_list = [day_rows[tau] for tau in range(di - w_days, di) if tau in day_rows]
                hist_idx = np.concatenate(hist_idx_list) if hist_idx_list else np.empty(0, dtype=np.int64)

                if len(hist_idx) >= min_pts:
                    q_roll = weighted_quantile(scores_sym[hist_idx], level)
                    ql_roll = weighted_quantile(scores_low[hist_idx], 1.0 - 0.5 * alpha)
                    qh_roll = weighted_quantile(scores_high[hist_idx], 1.0 - 0.5 * alpha)
                    q_aci = weighted_quantile(scores_sym[hist_idx], float(np.clip(1.0 - alpha_t, 0.50, 0.995)))
                else:
                    q_roll = q0
                    ql_roll, qh_roll = ql0, qh0
                    q_aci = q0

                if not is_alarm:
                    clean_list = [
                        day_rows[tau]
                        for tau in range(di - w_days, di)
                        if tau in day_rows and not alarms.get((s, tau), (False, 1.0))[0]
                    ]
                    clean_idx = np.concatenate(clean_list) if clean_list else np.empty(0, dtype=np.int64)
                    if len(clean_idx) >= min_pts:
                        q_fg_last = weighted_quantile(scores_sym[clean_idx], level)
                        ql_fg_last = weighted_quantile(scores_low[clean_idx], 1.0 - 0.5 * alpha)
                        qh_fg_last = weighted_quantile(scores_high[clean_idx], 1.0 - 0.5 * alpha)

                for m_name, (ql_use, qh_use) in (
                    ("Rolling", (q_roll, q_roll)),
                    ("Causal_ACI", (q_aci, q_aci)),
                    ("Asym_Rolling", (ql_roll, qh_roll)),
                    ("FaultGated_Rolling", (q_fg_last, q_fg_last)),
                    ("FaultGated_Asym", (ql_fg_last, qh_fg_last)),
                ):
                    lo = np.maximum(0.0, center_preds[idx] - ql_use * sigma[idx] * c[idx])
                    hi = center_preds[idx] + qh_use * sigma[idx] * c[idx]
                    out[m_name]["covered"][idx] = (split.power[idx] >= lo) & (split.power[idx] <= hi)
                    out[m_name]["width_normalized"][idx] = (hi - lo) / c[idx]

                err_day = float(np.mean(~out["Causal_ACI"]["covered"][idx]))
                alpha_t = float(np.clip(alpha_t + self.config.aci_step_size * (alpha - err_day), 0.005, 0.50))

        return out
