from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .solar import compute_solar_and_cyclic_features


@dataclass
class PreparedSplit:
    timestamps_ns: np.ndarray
    stream_ids: np.ndarray
    technology_codes: np.ndarray
    poa: np.ndarray
    air_c: np.ndarray
    solar_coszen: np.ndarray
    hour_sin: np.ndarray
    hour_cos: np.ndarray
    doy_sin: np.ndarray
    doy_cos: np.ndarray
    power: np.ndarray
    wind: np.ndarray
    wind_valid: np.ndarray
    reanalysis_wind: np.ndarray | None
    year: np.ndarray
    day_index: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(len(self.power))

    def wind_free_matrix(self, include_wind: np.ndarray | None = None) -> np.ndarray:
        cols = [
            self.poa / 1000.0,
            self.air_c,
            self.solar_coszen,
            self.hour_sin,
            self.hour_cos,
            self.doy_sin,
            self.doy_cos,
            self.technology_codes.astype(np.float64),
            self.stream_ids.astype(np.float64),
        ]
        if include_wind is not None:
            cols.append(np.asarray(include_wind, dtype=np.float64) / 10.0)
        return np.column_stack(cols)


@dataclass
class FleetScales:
    stream_scales: dict[int, float]
    stream_scales_subtrain: dict[int, float]
    technology_of_stream: dict[int, int]
    technology_names: dict[int, str]
    nameplate_kwp: dict[int, float]

    def scale_array(self, stream_ids: np.ndarray, subtrain: bool = False) -> np.ndarray:
        table = self.stream_scales_subtrain if subtrain else self.stream_scales
        return np.array([table[int(s)] for s in stream_ids], dtype=np.float64)

    def kwp_array(self, stream_ids: np.ndarray) -> np.ndarray:
        return np.array([self.nameplate_kwp.get(int(s), self.stream_scales[int(s)]) for s in stream_ids], dtype=np.float64)


def parse_timestamp_ns(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, utc=False)
    return dt.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def apply_service_windows(
    df: pd.DataFrame,
    timestamp_col: str,
    stream_id_col: str,
    exclusion_windows: list[dict],
) -> pd.DataFrame:
    if not exclusion_windows:
        return df
    ts = pd.to_datetime(df[timestamp_col], utc=False)
    sid = df[stream_id_col].astype(int)
    keep = np.ones(len(df), dtype=bool)
    for rule in exclusion_windows:
        target_id = int(rule["system_id"])
        m = (sid == target_id).to_numpy()
        if rule.get("exclude_before"):
            m = m & (ts < pd.Timestamp(rule["exclude_before"])).to_numpy()
        elif rule.get("exclude_after"):
            m = m & (ts > pd.Timestamp(rule["exclude_after"])).to_numpy()
        elif rule.get("start") and rule.get("end"):
            m = m & (ts >= pd.Timestamp(rule["start"])).to_numpy() & (ts <= pd.Timestamp(rule["end"])).to_numpy()
        keep &= ~m
    return df.loc[keep].reset_index(drop=True)


def compute_fleet_scales(
    train_split: PreparedSplit,
    subtrain_mask: np.ndarray,
    technology_names: dict[int, str],
    nameplate_kwp: dict[int, float],
    quantile: float = 0.99,
    min_rows: int = 500,
) -> FleetScales:
    unique_streams = sorted(int(s) for s in np.unique(train_split.stream_ids))
    tech_of_stream: dict[int, int] = {}
    raw_full: dict[int, float] = {}
    raw_sub: dict[int, float] = {}

    for s in unique_streams:
        m_full = (train_split.stream_ids == s) & np.isfinite(train_split.power) & (train_split.power >= 0.0)
        tech_of_stream[s] = int(train_split.technology_codes[m_full][0])
        if int(m_full.sum()) >= min_rows:
            raw_full[s] = float(np.quantile(train_split.power[m_full], quantile))
        m_sub = m_full & subtrain_mask
        if int(m_sub.sum()) >= min_rows:
            raw_sub[s] = float(np.quantile(train_split.power[m_sub], quantile))

    fleet_full_median = float(np.median(list(raw_full.values()))) if raw_full else 1.0
    fleet_sub_median = float(np.median(list(raw_sub.values()))) if raw_sub else fleet_full_median

    scales_full: dict[int, float] = {}
    scales_sub: dict[int, float] = {}
    for s in unique_streams:
        k = tech_of_stream[s]
        if s in raw_full and raw_full[s] > 0:
            scales_full[s] = raw_full[s]
        else:
            peer_vals = [v for peer, v in raw_full.items() if tech_of_stream[peer] == k and v > 0]
            scales_full[s] = float(np.median(peer_vals)) if peer_vals else fleet_full_median

        if s in raw_sub and raw_sub[s] > 0:
            scales_sub[s] = raw_sub[s]
        else:
            peer_sub = [v for peer, v in raw_sub.items() if tech_of_stream[peer] == k and v > 0]
            scales_sub[s] = float(np.median(peer_sub)) if peer_sub else fleet_sub_median

    return FleetScales(
        stream_scales=scales_full,
        stream_scales_subtrain=scales_sub,
        technology_of_stream=tech_of_stream,
        technology_names=technology_names,
        nameplate_kwp=nameplate_kwp,
    )


def prepare_dataframe_to_split(
    df: pd.DataFrame,
    cfg: BenchmarkConfig,
    technology_code_map: dict[str, int],
) -> PreparedSplit:
    sc = cfg.schema
    st = cfg.site
    poa = pd.to_numeric(df[sc.irradiance_col], errors="coerce").to_numpy(dtype=np.float64)
    air_c = pd.to_numeric(df[sc.ambient_temp_col], errors="coerce").to_numpy(dtype=np.float64)
    power = pd.to_numeric(df[sc.power_col], errors="coerce").to_numpy(dtype=np.float64)
    valid_env = (
        np.isfinite(poa)
        & (poa >= st.irradiance_min_wm2)
        & (poa < st.irradiance_max_wm2)
        & np.isfinite(air_c)
        & (air_c >= st.ambient_temp_min_c)
        & (air_c <= st.ambient_temp_max_c)
        & np.isfinite(power)
    )
    sub = df.loc[valid_env].reset_index(drop=True)
    ts_ns = parse_timestamp_ns(sub[sc.timestamp_col])
    solar = compute_solar_and_cyclic_features(ts_ns, st)
    stream_ids = sub[sc.stream_id_col].to_numpy(dtype=np.int32)
    if np.issubdtype(sub[sc.technology_col].dtype, np.number):
        tech_codes = sub[sc.technology_col].to_numpy(dtype=np.int32)
    else:
        tech_codes = np.array([technology_code_map[str(t)] for t in sub[sc.technology_col]], dtype=np.int32)

    if sc.wind_speed_col in sub.columns:
        w = pd.to_numeric(sub[sc.wind_speed_col], errors="coerce").to_numpy(dtype=np.float64)
        w_ok = np.isfinite(w) & (w >= st.wind_min_ms) & (w <= st.wind_max_ms)
        w_clean = np.where(w_ok, w, np.nan)
    else:
        w_clean = np.full(len(sub), np.nan, dtype=np.float64)
        w_ok = np.zeros(len(sub), dtype=bool)

    if sc.reanalysis_wind_col in sub.columns:
        rw = pd.to_numeric(sub[sc.reanalysis_wind_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        rw = None

    return PreparedSplit(
        timestamps_ns=ts_ns,
        stream_ids=stream_ids,
        technology_codes=tech_codes,
        poa=sub[sc.irradiance_col].to_numpy(dtype=np.float64),
        air_c=sub[sc.ambient_temp_col].to_numpy(dtype=np.float64),
        solar_coszen=solar["solar_coszen"].astype(np.float64),
        hour_sin=solar["hour_sin"].astype(np.float64),
        hour_cos=solar["hour_cos"].astype(np.float64),
        doy_sin=solar["doy_sin"].astype(np.float64),
        doy_cos=solar["doy_cos"].astype(np.float64),
        power=sub[sc.power_col].to_numpy(dtype=np.float64),
        wind=w_clean,
        wind_valid=w_ok,
        reanalysis_wind=rw,
        year=solar["year"],
        day_index=solar["day_index"],
    )


def subset_split(split: PreparedSplit, mask: np.ndarray) -> PreparedSplit:
    idx = np.asarray(mask, dtype=bool)
    return PreparedSplit(
        timestamps_ns=split.timestamps_ns[idx],
        stream_ids=split.stream_ids[idx],
        technology_codes=split.technology_codes[idx],
        poa=split.poa[idx],
        air_c=split.air_c[idx],
        solar_coszen=split.solar_coszen[idx],
        hour_sin=split.hour_sin[idx],
        hour_cos=split.hour_cos[idx],
        doy_sin=split.doy_sin[idx],
        doy_cos=split.doy_cos[idx],
        power=split.power[idx],
        wind=split.wind[idx],
        wind_valid=split.wind_valid[idx],
        reanalysis_wind=split.reanalysis_wind[idx] if split.reanalysis_wind is not None else None,
        year=split.year[idx],
        day_index=split.day_index[idx],
    )


def deterministic_Sample_by_stream(
    split: PreparedSplit,
    max_rows_per_stream: int | None,
    seed: int = 42,
    per_stream_seed_offset: int | None = 1729,
) -> np.ndarray:
    if max_rows_per_stream is None:
        return np.ones(split.n_rows, dtype=bool)
    shared_rng = np.random.default_rng(seed) if per_stream_seed_offset is None else None
    chosen = np.zeros(split.n_rows, dtype=bool)
    for s in sorted(int(x) for x in np.unique(split.stream_ids)):
        rows = np.flatnonzero(split.stream_ids == s)
        if len(rows) <= max_rows_per_stream:
            chosen[rows] = True
        else:
            rng = np.random.default_rng(per_stream_seed_offset + int(s)) if per_stream_seed_offset is not None else shared_rng
            pick = rng.choice(rows, size=max_rows_per_stream, replace=False)
            chosen[np.sort(pick)] = True
    return chosen


def load_json_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
