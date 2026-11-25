from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ThermalChainConfig, TreeConfig
from .data import FleetScales, PreparedSplit, sample_by_stream, subset_split
from .solar import build_harmonic_poa_basis


class WindFreeBoostingEstimator:
    def __init__(self, config: TreeConfig, use_categorical_ids: bool = False) -> None:
        self.config = config
        self.use_categorical_ids = use_categorical_ids
        self.model: HistGradientBoostingRegressor | None = None

    def _build_regressor(self) -> HistGradientBoostingRegressor:
        cat = [7, 8] if self.use_categorical_ids else None
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            categorical_features=cat,
            early_stopping=self.config.early_stopping,
            random_state=self.config.random_seed,
        )

    def fit(
        self,
        split: PreparedSplit,
        scales: FleetScales,
        use_subtrain_scales: bool = False,
        extra_wind: np.ndarray | None = None,
    ) -> WindFreeBoostingEstimator:
        mask = sample_by_stream(
            split,
            self.config.max_rows_per_stream,
            seed=self.config.random_seed,
            per_stream_seed_offset=self.config.per_stream_seed_offset,
        )
        sub = subset_split(split, mask)
        w_extra = extra_wind[mask] if extra_wind is not None else None
        x = sub.wind_free_matrix(include_wind=w_extra)
        c = scales.scale_array(sub.stream_ids, subtrain=use_subtrain_scales)
        y = np.clip(sub.power / c, 0.0, self.config.max_normalized_power)
        self.model = self._build_regressor()
        self.model.fit(x, y)
        return self

    def predict(
        self,
        split: PreparedSplit,
        scales: FleetScales,
        use_subtrain_scales: bool = False,
        extra_wind: np.ndarray | None = None,
        trend_rates: dict[int | str, float] | None = None,
        reference_year: float = 2014.5,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Estimator has not been fitted.")
        x = split.wind_free_matrix(include_wind=extra_wind)
        c = scales.scale_array(split.stream_ids, subtrain=use_subtrain_scales)
        y_norm = np.clip(self.model.predict(x), 0.0, self.config.max_normalized_power)
        pred = y_norm * c
        if trend_rates is not None:
            ts_years = 1970.0 + (split.timestamps_ns.astype(np.float64) / (1e9 * 86400.0 * 365.25))
            dt = np.maximum(0.0, ts_years - reference_year)
            rate_by_code = {
                int(k_code): float(
                    trend_rates[int(k_code)]
                    if int(k_code) in trend_rates
                    else trend_rates.get(scales.technology_names.get(int(k_code), ""), 0.0)
                )
                for k_code in np.unique(split.technology_codes)
            }
            rd = np.array([rate_by_code.get(int(k), 0.0) for k in split.technology_codes], dtype=np.float64)
            pred = pred * np.maximum(0.0, 1.0 - rd * dt)
        return pred


class PhysicalThermalChainEstimator:
    def __init__(self, config: ThermalChainConfig, wind_tree_config: TreeConfig) -> None:
        self.config = config
        self.wind_tree_config = wind_tree_config
        self.wind_imputer: HistGradientBoostingRegressor | None = None
        self.reanalysis_intercept: float = 0.0
        self.reanalysis_slope: float = 1.0
        self.power_coefs: dict[int, tuple[float, float]] = {}
        self.poa_transposition_weights: dict[int, np.ndarray] = {}

    @staticmethod
    def _wind_feature_matrix(split: PreparedSplit) -> np.ndarray:
        return split.wind_free_matrix()

    def fit_wind_imputer(self, train_split: PreparedSplit) -> PhysicalThermalChainEstimator:
        mw = train_split.wind_valid
        sub = subset_split(train_split, mw)
        x = self._wind_feature_matrix(sub)
        y = np.log1p(np.maximum(sub.wind, 0.0))
        self.wind_imputer = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.wind_tree_config.learning_rate,
            max_iter=self.wind_tree_config.max_iter,
            max_leaf_nodes=self.wind_tree_config.max_leaf_nodes,
            max_depth=self.wind_tree_config.max_depth,
            min_samples_leaf=self.wind_tree_config.min_samples_leaf,
            l2_regularization=self.wind_tree_config.l2_regularization,
            early_stopping=self.wind_tree_config.early_stopping,
            random_state=self.wind_tree_config.random_seed,
        )
        self.wind_imputer.fit(x, y)
        return self

    def impute_wind(self, split: PreparedSplit) -> np.ndarray:
        if self.wind_imputer is None:
            raise RuntimeError("Wind imputer has not been fitted.")
        x = self._wind_feature_matrix(split)
        return np.maximum(np.expm1(self.wind_imputer.predict(x)), 0.0)

    def fit_reanalysis_affine_calibration(self, train_split: PreparedSplit) -> tuple[float, float]:
        if train_split.reanalysis_wind is None:
            raise ValueError("Reanalysis wind series is required for affine calibration.")
        m = train_split.wind_valid & np.isfinite(train_split.reanalysis_wind)
        w_era = train_split.reanalysis_wind[m]
        w_obs = train_split.wind[m]
        x = np.column_stack([np.ones(len(w_era)), w_era])
        beta, _, _, _ = np.linalg.lstsq(x, w_obs, rcond=None)
        self.reanalysis_intercept = float(beta[0])
        self.reanalysis_slope = float(beta[1])
        return self.reanalysis_intercept, self.reanalysis_slope

    def calibrate_reanalysis_wind(self, reanalysis_wind: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.reanalysis_intercept + self.reanalysis_slope * reanalysis_wind)

    def _gamma_for_split(self, split: PreparedSplit, scales: FleetScales) -> np.ndarray:
        if not self.config.use_datasheet_gamma:
            return np.full(split.n_rows, self.config.default_gamma_per_c, dtype=np.float64)
        out = np.empty(split.n_rows, dtype=np.float64)
        for k_code, k_name in scales.technology_names.items():
            g_val = self.config.technology_gamma_per_c.get(k_name, self.config.default_gamma_per_c)
            out[split.technology_codes == k_code] = g_val
        return out

    def _thermal_factor(self, split: PreparedSplit, wind: np.ndarray, scales: FleetScales) -> np.ndarray:
        t_mod = split.air_c + split.poa / (self.config.faiman_u0 + self.config.faiman_u1 * np.maximum(wind, 0.0))
        gamma = self._gamma_for_split(split, scales)
        return 1.0 + gamma * (t_mod - self.config.reference_temp_c)

    def fit_power_chain(
        self,
        train_split: PreparedSplit,
        scales: FleetScales,
        wind_override: np.ndarray | None = None,
    ) -> PhysicalThermalChainEstimator:
        w = wind_override if wind_override is not None else self.impute_wind(train_split)
        tf = self._thermal_factor(train_split, w, scales)
        c = scales.scale_array(train_split.stream_ids)
        y_norm = train_split.power / c

        if self.config.use_harmonic_poa_transposition:
            basis = build_harmonic_poa_basis(
                train_split.poa,
                train_split.solar_coszen,
                train_split.hour_sin,
                train_split.hour_cos,
                train_split.doy_sin,
                train_split.doy_cos,
                reference_irradiance_wm2=self.config.reference_irradiance_wm2,
            )
            p = basis.shape[1]
            for s in sorted(int(v) for v in np.unique(train_split.stream_ids)):
                m = train_split.stream_ids == s
                xb = basis[m] * tf[m, None]
                x_aug = np.column_stack([np.ones(int(np.sum(m))), xb])
                reg = self.config.transposition_ridge_lambda * np.eye(p + 1)
                reg[0, 0] = 0.0
                theta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_norm[m])
                self.poa_transposition_weights[s] = theta
        else:
            z = (train_split.poa / self.config.reference_irradiance_wm2) * tf
            ridge = self.config.response_ridge_scale
            for s in sorted(int(v) for v in np.unique(train_split.stream_ids)):
                m = train_split.stream_ids == s
                xx = z[m]
                yy = y_norm[m]
                n = float(len(xx))
                lhs = np.array(
                    [
                        [n * (1.0 + ridge), float(np.sum(xx))],
                        [float(np.sum(xx)), float(np.dot(xx, xx)) + n * ridge],
                    ],
                    dtype=np.float64,
                )
                rhs = np.array([float(np.sum(yy)), float(np.dot(xx, yy)) + n * ridge], dtype=np.float64)
                beta = np.linalg.solve(lhs, rhs)
                self.power_coefs[s] = (float(beta[0]), float(beta[1]))
        return self

    def predict(
        self,
        split: PreparedSplit,
        scales: FleetScales,
        wind_override: np.ndarray | None = None,
    ) -> np.ndarray:
        w = wind_override if wind_override is not None else self.impute_wind(split)
        tf = self._thermal_factor(split, w, scales)
        c = scales.scale_array(split.stream_ids)
        y_norm = np.zeros(split.n_rows, dtype=np.float64)

        if self.config.use_harmonic_poa_transposition:
            basis = build_harmonic_poa_basis(
                split.poa,
                split.solar_coszen,
                split.hour_sin,
                split.hour_cos,
                split.doy_sin,
                split.doy_cos,
                reference_irradiance_wm2=self.config.reference_irradiance_wm2,
            )
            default_theta = (
                np.mean(list(self.poa_transposition_weights.values()), axis=0)
                if self.poa_transposition_weights
                else np.zeros(basis.shape[1] + 1)
            )
            for s in sorted(int(v) for v in np.unique(split.stream_ids)):
                m = split.stream_ids == s
                theta = self.poa_transposition_weights.get(s, default_theta)
                xb = basis[m] * tf[m, None]
                y_norm[m] = theta[0] + xb @ theta[1:]
        else:
            z = (split.poa / self.config.reference_irradiance_wm2) * tf
            default_ab = (
                (
                    float(np.mean([v[0] for v in self.power_coefs.values()])),
                    float(np.mean([v[1] for v in self.power_coefs.values()])),
                )
                if self.power_coefs
                else (0.0, 1.0)
            )
            for s in sorted(int(v) for v in np.unique(split.stream_ids)):
                m = split.stream_ids == s
                a_s, b_s = self.power_coefs.get(s, default_ab)
                y_norm[m] = a_s + b_s * z[m]

        if self.config.clip_predictions:
            y_norm = np.clip(y_norm, 0.0, self.config.max_normalized_power)
        else:
            y_norm = np.maximum(0.0, y_norm)
        return y_norm * c
