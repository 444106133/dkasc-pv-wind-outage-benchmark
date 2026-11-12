from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear, minimize
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import (
    FallbackHybridConfig,
    GateAndBootstrapConfig,
    MixtureOfExpertsConfig,
    ThermalChainConfig,
    TreeConfig,
)
from .data import FleetScales, PreparedSplit, deterministic_Sample_by_stream, subset_split
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
        mask = deterministic_Sample_by_stream(
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
        degradation_rates: dict[int | str, float] | None = None,
        reference_year: float = 2014.5,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Estimator has not been fitted.")
        x = split.wind_free_matrix(include_wind=extra_wind)
        c = scales.scale_array(split.stream_ids, subtrain=use_subtrain_scales)
        y_norm = np.clip(self.model.predict(x), 0.0, self.config.max_normalized_power)
        pred = y_norm * c
        if degradation_rates is not None:
            ts_years = 1970.0 + (split.timestamps_ns.astype(np.float64) / (1e9 * 86400.0 * 365.25))
            dt = np.maximum(0.0, ts_years - reference_year)
            rate_by_code = {
                int(k_code): float(
                    degradation_rates[int(k_code)]
                    if int(k_code) in degradation_rates
                    else degradation_rates.get(scales.technology_names.get(int(k_code), ""), 0.0)
                )
                for k_code in np.unique(split.technology_codes)
            }
            rd = np.array([rate_by_code.get(int(k), 0.0) for k in split.technology_codes], dtype=np.float64)
            pred = pred * np.maximum(0.0, 1.0 - rd * dt)
        return pred


class MeasuredWindFallbackHybrid:
    def __init__(self, config: FallbackHybridConfig) -> None:
        self.config = config
        self.beta_phys: dict[int, np.ndarray] = {}
        self.core_fleet: np.ndarray = np.zeros(8, dtype=np.float64)
        self.core_tech: dict[int, np.ndarray] = {}
        self.core_stream: dict[int, np.ndarray] = {}
        self.wind_fleet: np.ndarray = np.zeros(4, dtype=np.float64)
        self.wind_stream: dict[int, np.ndarray] = {}

    @staticmethod
    def _phys_basis(split: PreparedSplit) -> np.ndarray:
        g = split.poa / 1000.0
        dt = (split.air_c - 25.0) / 100.0
        return np.column_stack([g, -g * np.maximum(dt, 0.0)])

    @staticmethod
    def _core_basis(split: PreparedSplit) -> np.ndarray:
        g = split.poa / 1000.0
        t = (split.air_c - 25.0) / 25.0
        z = split.solar_coszen
        return np.column_stack([g, g * g, g * t, t, z, g * z, split.hour_sin, split.hour_cos])

    @staticmethod
    def _wind_basis(split: PreparedSplit, wind: np.ndarray) -> np.ndarray:
        g = split.poa / 1000.0
        t = (split.air_c - 25.0) / 25.0
        w = (wind - 3.0) / 3.0
        return np.column_stack([w, g * w, g * t * w, np.tanh(w)])

    @staticmethod
    def _solve_ridge(x: np.ndarray, y: np.ndarray, lam: float, prior: np.ndarray, clip: float) -> np.ndarray:
        p = x.shape[1]
        lhs = x.T @ x + lam * np.eye(p)
        rhs = x.T @ y + lam * prior
        coef = np.linalg.solve(lhs, rhs)
        return np.clip(coef, -clip, clip)

    def fit(self, split: PreparedSplit, scales: FleetScales) -> MeasuredWindFallbackHybrid:
        mask = deterministic_Sample_by_stream(split, self.config.max_rows_per_stream, seed=self.config.random_seed)
        sub = subset_split(split, mask)
        c = scales.scale_array(sub.stream_ids)
        y = np.clip(sub.power / c, self.config.target_clip_low, self.config.target_clip_high)

        xp = self._phys_basis(sub)
        phys_pred = np.zeros(sub.n_rows, dtype=np.float64)
        lam_sqrt = np.sqrt(self.config.phys_ridge_lambda)
        lb = [self.config.phys_bounds_b1[0], self.config.phys_bounds_b2[0]]
        ub = [self.config.phys_bounds_b1[1], self.config.phys_bounds_b2[1]]

        for s in sorted(int(v) for v in np.unique(sub.stream_ids)):
            m = sub.stream_ids == s
            aug_x = np.vstack([xp[m], lam_sqrt * np.eye(2)])
            aug_y = np.concatenate([y[m], np.zeros(2)])
            res = lsq_linear(aug_x, aug_y, bounds=(lb, ub), tol=1e-8)
            self.beta_phys[s] = res.x
            phys_pred[m] = xp[m] @ res.x

        xc = self._core_basis(sub)
        r_core = y - phys_pred
        self.core_fleet = self._solve_ridge(
            xc,
            r_core,
            self.config.core_fleet_lambda,
            np.zeros(xc.shape[1]),
            self.config.core_coef_clip,
        )

        streams_by_tech: dict[int, set[int]] = {}
        for s, k in scales.technology_of_stream.items():
            streams_by_tech.setdefault(k, set()).add(s)

        for k in sorted(streams_by_tech):
            mk = sub.technology_codes == k
            if np.any(mk):
                raw_k = self._solve_ridge(
                    xc[mk],
                    r_core[mk],
                    self.config.core_tech_lambda,
                    self.core_fleet,
                    self.config.core_coef_clip,
                )
                nk = len(streams_by_tech[k])
                w_fleet = nk / (nk + 4.0)
                self.core_tech[k] = w_fleet * raw_k + (1.0 - w_fleet) * self.core_fleet
            else:
                self.core_tech[k] = self.core_fleet.copy()

        core_pred = np.zeros(sub.n_rows, dtype=np.float64)
        for s in sorted(int(v) for v in np.unique(sub.stream_ids)):
            m = sub.stream_ids == s
            k = scales.technology_of_stream[s]
            self.core_stream[s] = self._solve_ridge(
                xc[m],
                r_core[m],
                self.config.core_stream_lambda,
                self.core_tech[k],
                self.config.core_coef_clip,
            )
            core_pred[m] = np.clip(
                phys_pred[m] + xc[m] @ self.core_stream[s],
                self.config.target_clip_low,
                self.config.target_clip_high,
            )

        mw = sub.wind_valid
        if np.any(mw):
            xw = self._wind_basis(subset_split(sub, mw), sub.wind[mw])
            rw = y[mw] - core_pred[mw]
            self.wind_fleet = self._solve_ridge(
                xw,
                rw,
                self.config.wind_fleet_lambda,
                np.zeros(xw.shape[1]),
                self.config.wind_coef_clip,
            )
            for s in sorted(int(v) for v in np.unique(sub.stream_ids)):
                ms = mw & (sub.stream_ids == s)
                if np.any(ms):
                    xws = self._wind_basis(subset_split(sub, ms), sub.wind[ms])
                    self.wind_stream[s] = self._solve_ridge(
                        xws,
                        y[ms] - core_pred[ms],
                        self.config.wind_stream_lambda,
                        self.wind_fleet,
                        self.config.wind_coef_clip,
                    )
                else:
                    self.wind_stream[s] = self.wind_fleet.copy()
        return self

    def predict(self, split: PreparedSplit, scales: FleetScales, allow_wind: bool = True) -> np.ndarray:
        xp = self._phys_basis(split)
        xc = self._core_basis(split)
        out = np.zeros(split.n_rows, dtype=np.float64)
        default_phys = np.mean(list(self.beta_phys.values()), axis=0) if self.beta_phys else np.array([1.0, 0.1])

        for s in sorted(int(v) for v in np.unique(split.stream_ids)):
            m = split.stream_ids == s
            bp = self.beta_phys.get(s, default_phys)
            bc = self.core_stream.get(s, self.core_fleet)
            base = np.clip(
                xp[m] @ bp + xc[m] @ bc,
                self.config.target_clip_low,
                self.config.target_clip_high,
            )
            if allow_wind:
                mw = m & split.wind_valid
                if np.any(mw):
                    bw = self.wind_stream.get(s, self.wind_fleet)
                    xw = self._wind_basis(subset_split(split, mw), split.wind[mw])
                    idx_local = np.flatnonzero(split.wind_valid[m])
                    base[idx_local] = np.clip(
                        base[idx_local] + xw @ bw,
                        self.config.target_clip_low,
                        self.config.target_clip_high,
                    )
            out[m] = base

        c = scales.scale_array(split.stream_ids)
        return np.maximum(0.0, out * c)


class AvailabilityGatedMixtureOfExperts:
    def __init__(self, config: MixtureOfExpertsConfig, tree_config: TreeConfig) -> None:
        self.config = config
        self.tree_config = tree_config
        self.phys_expert: MeasuredWindFallbackHybrid | None = None
        self.wf_expert: WindFreeBoostingEstimator | None = None
        self.wind_expert: WindFreeBoostingEstimator | None = None
        self.weights_no_wind: dict[int, np.ndarray] = {}
        self.weights_with_wind: dict[int, np.ndarray] = {}
        self.pooled_no_wind: np.ndarray = np.array(config.prior_no_wind, dtype=np.float64)
        self.pooled_with_wind: np.ndarray = np.array(config.prior_with_wind, dtype=np.float64)

    def _regime_ids(self, split: PreparedSplit) -> np.ndarray:
        e1, e2 = self.config.irradiance_edges
        g_bin = np.where(split.poa < e1, 0, np.where(split.poa < e2, 1, 2))
        z_bin = np.where(split.solar_coszen < self.config.coszen_edge, 0, 1)
        return (g_bin * 2 + z_bin).astype(np.int32)

    @staticmethod
    def _fit_simplex_weights(preds: np.ndarray, target: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
        k = preds.shape[1]

        def obj(w: np.ndarray) -> float:
            r = preds @ w - target
            return float(np.dot(r, r) + lam * np.dot(w - prior, w - prior))

        cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        bounds = [(0.0, 1.0)] * k
        res = minimize(obj, prior.copy(), method="SLSQP", bounds=bounds, constraints=cons)
        w = np.clip(res.x if res.success else prior, 0.0, 1.0)
        return w / np.sum(w)

    def fit(
        self,
        train_split: PreparedSplit,
        subtrain_mask: np.ndarray,
        forward_2015_mask: np.ndarray,
        scales: FleetScales,
    ) -> AvailabilityGatedMixtureOfExperts:
        sub_split = subset_split(train_split, subtrain_mask)
        fwd_split = subset_split(train_split, forward_2015_mask & train_split.wind_valid)
        fwd_sample_mask = deterministic_Sample_by_stream(
            fwd_split,
            self.config.forward_rows_per_stream,
            seed=self.config.random_seed,
        )
        fwd_sample = subset_split(fwd_split, fwd_sample_mask)

        fb_cfg = FallbackHybridConfig(random_seed=self.config.random_seed)
        phys_sub = MeasuredWindFallbackHybrid(fb_cfg).fit(sub_split, scales)
        wf_sub = WindFreeBoostingEstimator(self.tree_config, use_categorical_ids=True).fit(
            sub_split,
            scales,
            use_subtrain_scales=True,
        )
        sub_wind = subset_split(sub_split, sub_split.wind_valid)
        wind_sub = WindFreeBoostingEstimator(self.tree_config, use_categorical_ids=True).fit(
            sub_wind,
            scales,
            use_subtrain_scales=True,
            extra_wind=sub_wind.wind,
        )

        c_fwd = scales.scale_array(fwd_sample.stream_ids, subtrain=True)
        y_fwd = fwd_sample.power / c_fwd
        p_phys = phys_sub.predict(fwd_sample, scales, allow_wind=False) / c_fwd
        p_wf = wf_sub.predict(fwd_sample, scales, use_subtrain_scales=True) / c_fwd
        p_w = wind_sub.predict(fwd_sample, scales, use_subtrain_scales=True, extra_wind=fwd_sample.wind) / c_fwd

        mat_0 = np.column_stack([p_phys, p_wf])
        mat_1 = np.column_stack([p_phys, p_wf, p_w])
        prior_0 = np.array(self.config.prior_no_wind, dtype=np.float64)
        prior_1 = np.array(self.config.prior_with_wind, dtype=np.float64)

        self.pooled_no_wind = self._fit_simplex_weights(mat_0, y_fwd, prior_0, self.config.pooled_lambda)
        self.pooled_with_wind = self._fit_simplex_weights(mat_1, y_fwd, prior_1, self.config.pooled_lambda)

        regimes = self._regime_ids(fwd_sample)
        for b in range(6):
            mb = regimes == b
            nb = int(np.sum(mb))
            if nb >= self.config.min_regime_rows:
                w0_hat = self._fit_simplex_weights(mat_0[mb], y_fwd[mb], self.pooled_no_wind, self.config.regime_lambda)
                w1_hat = self._fit_simplex_weights(mat_1[mb], y_fwd[mb], self.pooled_with_wind, self.config.regime_lambda)
                alpha = nb / (nb + self.config.shrinkage_pseudo_rows)
                self.weights_no_wind[b] = (1.0 - alpha) * self.pooled_no_wind + alpha * w0_hat
                self.weights_with_wind[b] = (1.0 - alpha) * self.pooled_with_wind + alpha * w1_hat
            else:
                self.weights_no_wind[b] = self.pooled_no_wind.copy()
                self.weights_with_wind[b] = self.pooled_with_wind.copy()

        self.phys_expert = MeasuredWindFallbackHybrid(fb_cfg).fit(train_split, scales)
        self.wf_expert = WindFreeBoostingEstimator(self.tree_config, use_categorical_ids=True).fit(
            train_split,
            scales,
            use_subtrain_scales=False,
        )
        full_wind = subset_split(train_split, train_split.wind_valid)
        self.wind_expert = WindFreeBoostingEstimator(self.tree_config, use_categorical_ids=True).fit(
            full_wind,
            scales,
            use_subtrain_scales=False,
            extra_wind=full_wind.wind,
        )
        return self

    def predict(self, split: PreparedSplit, scales: FleetScales, allow_wind: bool = True) -> np.ndarray:
        if self.phys_expert is None or self.wf_expert is None or self.wind_expert is None:
            raise RuntimeError("Mixture of experts has not been fitted.")
        p_phys = self.phys_expert.predict(split, scales, allow_wind=False)
        p_wf = self.wf_expert.predict(split, scales, use_subtrain_scales=False)
        regimes = self._regime_ids(split)
        out = np.zeros(split.n_rows, dtype=np.float64)

        has_w = split.wind_valid if allow_wind else np.zeros(split.n_rows, dtype=bool)
        if np.any(has_w):
            sub_w = subset_split(split, has_w)
            p_w_sub = self.wind_expert.predict(sub_w, scales, use_subtrain_scales=False, extra_wind=sub_w.wind)
            p_w = np.zeros(split.n_rows, dtype=np.float64)
            p_w[has_w] = p_w_sub
        else:
            p_w = np.zeros(split.n_rows, dtype=np.float64)

        for b in range(6):
            mb0 = (regimes == b) & (~has_w)
            if np.any(mb0):
                w0 = self.weights_no_wind.get(b, self.pooled_no_wind)
                out[mb0] = w0[0] * p_phys[mb0] + w0[1] * p_wf[mb0]
            mb1 = (regimes == b) & has_w
            if np.any(mb1):
                w1 = self.weights_with_wind.get(b, self.pooled_with_wind)
                out[mb1] = w1[0] * p_phys[mb1] + w1[1] * p_wf[mb1] + w1[2] * p_w[mb1]

        c = scales.scale_array(split.stream_ids)
        return np.clip(out, 0.0, self.tree_config.max_normalized_power * c)


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
        tc = split.air_c + split.poa / (self.config.faiman_u0 + self.config.faiman_u1 * np.maximum(wind, 0.0))
        gamma = self._gamma_for_split(split, scales)
        return 1.0 + gamma * (tc - self.config.reference_temp_c)

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


class WindValueDistillationGate:
    def __init__(self, tree_config: TreeConfig, gate_config: GateAndBootstrapConfig) -> None:
        self.tree_config = tree_config
        self.gate_config = gate_config
        self.wind_free_model: WindFreeBoostingEstimator | None = None
        self.teacher_model: WindFreeBoostingEstimator | None = None

    def fit_and_evaluate_gate(
        self,
        train_split: PreparedSplit,
        subtrain_mask: np.ndarray,
        forward_2015_mask: np.ndarray,
        scales: FleetScales,
    ) -> dict[str, float | int | bool | list[float]]:
        sub_split = subset_split(train_split, subtrain_mask & train_split.wind_valid)
        fwd_split = subset_split(train_split, forward_2015_mask & train_split.wind_valid)

        self.wind_free_model = WindFreeBoostingEstimator(self.tree_config).fit(
            sub_split,
            scales,
            use_subtrain_scales=True,
        )
        self.teacher_model = WindFreeBoostingEstimator(self.tree_config).fit(
            sub_split,
            scales,
            use_subtrain_scales=True,
            extra_wind=sub_split.wind,
        )

        pred_e = self.wind_free_model.predict(fwd_split, scales, use_subtrain_scales=True)
        pred_t = self.teacher_model.predict(fwd_split, scales, use_subtrain_scales=True, extra_wind=fwd_split.wind)
        c_sub = scales.scale_array(fwd_split.stream_ids, subtrain=True)

        err_e = np.abs(pred_e - fwd_split.power) / c_sub
        err_t = np.abs(pred_t - fwd_split.power) / c_sub

        streams = sorted(int(s) for s in np.unique(fwd_split.stream_ids))
        loss_e = float(np.mean([np.mean(err_e[fwd_split.stream_ids == s]) for s in streams]))
        loss_t = float(np.mean([np.mean(err_t[fwd_split.stream_ids == s]) for s in streams]))
        rel_gain = (loss_e - loss_t) / loss_e

        unique_days = np.sort(np.unique(fwd_split.day_index))
        block_len = self.gate_config.block_length_days
        n_blocks = max(1, int(np.ceil(len(unique_days) / block_len)))
        day_to_block = {int(d): int(i // block_len) for i, d in enumerate(unique_days)}
        row_blocks = np.array([day_to_block[int(d)] for d in fwd_split.day_index], dtype=np.int32)

        stream_block_sum_e = np.zeros((len(streams), n_blocks), dtype=np.float64)
        stream_block_sum_t = np.zeros((len(streams), n_blocks), dtype=np.float64)
        stream_block_cnt = np.zeros((len(streams), n_blocks), dtype=np.float64)

        for si, s in enumerate(streams):
            ms = fwd_split.stream_ids == s
            np.add.at(stream_block_sum_e[si], row_blocks[ms], err_e[ms])
            np.add.at(stream_block_sum_t[si], row_blocks[ms], err_t[ms])
            np.add.at(stream_block_cnt[si], row_blocks[ms], 1.0)

        rng = np.random.default_rng(self.gate_config.random_seed)
        draws: list[float] = []
        for _ in range(self.gate_config.n_bootstrap_draws):
            b_idx = rng.integers(0, n_blocks, size=n_blocks)
            cnt = np.sum(stream_block_cnt[:, b_idx], axis=1)
            ok = cnt > 0
            if np.any(ok):
                le_b = float(np.mean(np.sum(stream_block_sum_e[ok][:, b_idx], axis=1) / cnt[ok]))
                lt_b = float(np.mean(np.sum(stream_block_sum_t[ok][:, b_idx], axis=1) / cnt[ok]))
                draws.append((le_b - lt_b) / le_b)

        ci_low = float(np.quantile(draws, 0.025))
        ci_high = float(np.quantile(draws, 0.975))
        passed = bool((rel_gain >= self.gate_config.min_relative_loss_reduction) and (ci_low > 0.0))

        return {
            "paired_rows": int(fwd_split.n_rows),
            "n_streams": len(streams),
            "equal_system_wind_free_loss": loss_e,
            "equal_system_teacher_loss": loss_t,
            "relative_reduction": float(rel_gain),
            "ci95": [ci_low, ci_high],
            "gate_passed": passed,
        }
