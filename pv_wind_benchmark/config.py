from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SiteConfig:
    latitude_deg: float = -23.7618
    longitude_deg: float = 133.8748
    utc_offset_hours: float = 9.5
    irradiance_min_wm2: float = 50.0
    irradiance_max_wm2: float = 1400.0
    ambient_temp_min_c: float = -40.0
    ambient_temp_max_c: float = 60.0
    wind_min_ms: float = 0.0
    wind_max_ms: float = 50.0


@dataclass(frozen=True)
class DatasetSchema:
    timestamp_col: str = "timestamp"
    stream_id_col: str = "stream_id"
    technology_col: str = "technology"
    power_col: str = "active_power"
    irradiance_col: str = "poa_irradiance"
    ambient_temp_col: str = "ambient_temp_c"
    wind_speed_col: str = "wind_speed_ms"
    reanalysis_wind_col: str = "reanalysis_wind_ms"
    nameplate_kwp_col: str = "nameplate_kwp"


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2013-01-01T00:00:00"
    subtrain_end: str = "2014-12-31T23:59:59"
    calibration_split_timestamp: str = "2015-07-01T00:00:00"
    train_end: str = "2015-12-31T23:59:59"
    transition_end: str = "2016-12-31T23:59:59"
    wind_outage_timestamp: str = "2016-10-21T13:05:00"
    eval_end: str = "2026-01-20T12:10:00"
    scale_quantile: float = 0.99
    min_train_rows_for_stream_scale: int = 500


@dataclass(frozen=True)
class TreeConfig:
    max_iter: int = 180
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 120
    max_depth: int | None = None
    learning_rate: float = 0.06
    l2_regularization: float = 5.0
    max_rows_per_stream: int | None = None
    max_normalized_power: float = 1.3
    random_seed: int = 42
    early_stopping: bool = False
    per_stream_seed_offset: int | None = None


@dataclass(frozen=True)
class FallbackHybridConfig:
    max_rows_per_stream: int = 8000
    target_clip_low: float = -0.25
    target_clip_high: float = 1.50
    phys_bounds_b1: tuple[float, float] = (0.0, 1.6)
    phys_bounds_b2: tuple[float, float] = (0.0, 0.6)
    phys_ridge_lambda: float = 30.0
    core_fleet_lambda: float = 6000.0
    core_tech_lambda: float = 12000.0
    core_stream_lambda: float = 3000.0
    core_coef_clip: float = 0.15
    wind_fleet_lambda: float = 10000.0
    wind_stream_lambda: float = 3000.0
    wind_coef_clip: float = 0.08
    random_seed: int = 42


@dataclass(frozen=True)
class MixtureOfExpertsConfig:
    irradiance_edges: tuple[float, float] = (300.0, 700.0)
    coszen_edge: float = 0.45
    prior_no_wind: tuple[float, float] = (0.30, 0.70)
    prior_with_wind: tuple[float, float, float] = (0.20, 0.45, 0.35)
    pooled_lambda: float = 2000.0
    regime_lambda: float = 12000.0
    min_regime_rows: int = 500
    shrinkage_pseudo_rows: float = 12000.0
    forward_rows_per_stream: int = 2000
    random_seed: int = 42


@dataclass(frozen=True)
class ThermalChainConfig:
    faiman_u0: float = 25.0
    faiman_u1: float = 6.84
    reference_temp_c: float = 25.0
    reference_irradiance_wm2: float = 1000.0
    default_gamma_per_c: float = -0.004
    technology_gamma_per_c: dict[str, float] = field(
        default_factory=lambda: {
            "poly-Si": -0.0045,
            "mono-Si": -0.0040,
            "HIT": -0.0029,
            "CdTe": -0.0025,
            "CIGS": -0.0036,
            "a-Si": -0.0020,
            "CPV": -0.0015,
        }
    )
    technology_degradation_per_year: dict[str, float] = field(
        default_factory=lambda: {
            "poly-Si": 0.006,
            "mono-Si": 0.006,
            "HIT": 0.005,
            "CdTe": 0.005,
            "CIGS": 0.008,
            "a-Si": 0.008,
            "CPV": 0.008,
        }
    )
    use_datasheet_gamma: bool = False
    use_harmonic_poa_transposition: bool = False
    clip_predictions: bool = False
    max_normalized_power: float = 1.3
    transposition_ridge_lambda: float = 10.0
    response_ridge_scale: float = 0.01
    wind_imputer: TreeConfig = field(
        default_factory=lambda: TreeConfig(
            max_iter=60,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            max_depth=5,
            learning_rate=0.07,
            l2_regularization=20.0,
            max_rows_per_stream=None,
            max_normalized_power=1.3,
            random_seed=42,
            early_stopping=False,
            per_stream_seed_offset=None,
        )
    )


@dataclass(frozen=True)
class ConformalConfig:
    nominal_levels: tuple[float, ...] = (0.80, 0.90, 0.95)
    retention_width_quantile: float = 0.80
    min_stream_calibration_rows: int = 50
    rolling_window_days: int = 30
    min_rolling_residuals: int = 100
    aci_step_size: float = 0.05
    fault_gate_trailing_days: int = 7
    fault_gate_min_expected_fraction: float = 0.20
    fault_gate_min_high_irradiance_points: int = 30
    fault_gate_yield_threshold: float = 0.80
    scale_floor: float = 0.01
    scale_tree: TreeConfig = field(
        default_factory=lambda: TreeConfig(
            max_iter=80,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            max_depth=4,
            learning_rate=0.07,
            l2_regularization=10.0,
            max_rows_per_stream=None,
            max_normalized_power=1.3,
            random_seed=42,
            early_stopping=False,
            per_stream_seed_offset=None,
        )
    )


@dataclass(frozen=True)
class GateAndBootstrapConfig:
    min_relative_loss_reduction: float = 0.05
    block_length_days: int = 7
    n_bootstrap_draws: int = 2000
    random_seed: int = 42


@dataclass(frozen=True)
class BenchmarkConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    splits: SplitConfig = field(default_factory=SplitConfig)
    baseline_tree: TreeConfig = field(
        default_factory=lambda: TreeConfig(
            max_iter=110,
            max_leaf_nodes=20,
            min_samples_leaf=60,
            max_depth=None,
            learning_rate=0.08,
            l2_regularization=2.0,
            max_rows_per_stream=8000,
            max_normalized_power=1.3,
            random_seed=42,
            early_stopping=False,
            per_stream_seed_offset=1729,
        )
    )
    tuned_tree: TreeConfig = field(default_factory=TreeConfig)
    fallback_hybrid: FallbackHybridConfig = field(default_factory=FallbackHybridConfig)
    mixture_of_experts: MixtureOfExpertsConfig = field(default_factory=MixtureOfExpertsConfig)
    thermal_chain: ThermalChainConfig = field(default_factory=ThermalChainConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    inference: GateAndBootstrapConfig = field(default_factory=GateAndBootstrapConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> BenchmarkConfig:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        fb_raw = dict(raw.get("fallback_hybrid", {}))
        for k in ("phys_bounds_b1", "phys_bounds_b2"):
            if k in fb_raw:
                fb_raw[k] = tuple(float(v) for v in fb_raw[k])
        moe_raw = dict(raw.get("mixture_of_experts", {}))
        for k in ("irradiance_edges", "prior_no_wind", "prior_with_wind"):
            if k in moe_raw:
                moe_raw[k] = tuple(float(v) for v in moe_raw[k])
        tc_raw = dict(raw.get("thermal_chain", {}))
        wi_raw = tc_raw.pop("wind_imputer", None)
        tc_cfg = ThermalChainConfig(
            **tc_raw,
            wind_imputer=TreeConfig(**wi_raw) if isinstance(wi_raw, dict) else ThermalChainConfig().wind_imputer,
        )
        conf_raw = dict(raw.get("conformal", {}))
        st_raw = conf_raw.pop("scale_tree", None)
        if "nominal_levels" in conf_raw:
            conf_raw["nominal_levels"] = tuple(float(v) for v in conf_raw["nominal_levels"])
        conf_cfg = ConformalConfig(
            **conf_raw,
            scale_tree=TreeConfig(**st_raw) if isinstance(st_raw, dict) else ConformalConfig().scale_tree,
        )
        return cls(
            site=SiteConfig(**raw.get("site", {})),
            schema=DatasetSchema(**raw.get("schema", {})),
            splits=SplitConfig(**raw.get("splits", {})),
            baseline_tree=TreeConfig(**raw.get("baseline_tree", {})),
            tuned_tree=TreeConfig(**raw.get("tuned_tree", {})),
            fallback_hybrid=FallbackHybridConfig(**fb_raw),
            mixture_of_experts=MixtureOfExpertsConfig(**moe_raw),
            thermal_chain=tc_cfg,
            conformal=conf_cfg,
            inference=GateAndBootstrapConfig(**raw.get("inference", {})),
        )
