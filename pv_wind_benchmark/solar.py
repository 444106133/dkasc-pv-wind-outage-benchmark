from __future__ import annotations

import numpy as np

from .config import SiteConfig


def compute_solar_and_cyclic_features(
    timestamps_ns: np.ndarray,
    site: SiteConfig,
) -> dict[str, np.ndarray]:
    ts_sec = timestamps_ns.astype(np.int64) // 1_000_000_000
    dt_days = ts_sec // 86400
    sec_of_day = ts_sec - dt_days * 86400
    hour = sec_of_day.astype(np.float64) / 3600.0

    z = dt_days + 719468
    era = np.where(z >= 0, z, z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy0 = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy0 + 2) // 153
    d = doy0 - (153 * mp + 2) // 5 + 1
    m = mp + np.where(mp < 10, 3, -9)
    year = y + np.where(m <= 2, 1, 0)

    is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    month_offsets = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64)
    doy = month_offsets[m - 1] + d + np.where(is_leap & (m > 2), 1, 0)
    doy_f = doy.astype(np.float64)

    hour_angle_rad = 2.0 * np.pi * hour / 24.0
    doy_angle_rad = 2.0 * np.pi * doy_f / 365.25
    hour_sin = np.sin(hour_angle_rad)
    hour_cos = np.cos(hour_angle_rad)
    doy_sin = np.sin(doy_angle_rad)
    doy_cos = np.cos(doy_angle_rad)

    gamma = 2.0 * np.pi / 365.0 * (doy_f - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.001480 * np.sin(3.0 * gamma)
    )
    time_offset = eqtime + 4.0 * site.longitude_deg - 60.0 * site.utc_offset_hours
    tst = hour * 60.0 + time_offset
    ha_deg = tst / 4.0 - 180.0
    ha_rad = np.deg2rad(ha_deg)
    lat_rad = np.deg2rad(site.latitude_deg)

    coszen = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha_rad)
    solar_coszen = np.clip(coszen, 0.0, 1.0)

    return {
        "hour_sin": hour_sin.astype(np.float32),
        "hour_cos": hour_cos.astype(np.float32),
        "doy_sin": doy_sin.astype(np.float32),
        "doy_cos": doy_cos.astype(np.float32),
        "solar_coszen": solar_coszen.astype(np.float32),
        "year": year.astype(np.int32),
        "day_index": dt_days.astype(np.int64),
    }


def build_harmonic_poa_basis(
    irradiance_wm2: np.ndarray,
    solar_coszen: np.ndarray,
    hour_sin: np.ndarray,
    hour_cos: np.ndarray,
    doy_sin: np.ndarray,
    doy_cos: np.ndarray,
    reference_irradiance_wm2: float = 1000.0,
) -> np.ndarray:
    g = irradiance_wm2.astype(np.float64) / reference_irradiance_wm2
    z = solar_coszen.astype(np.float64)
    hs = hour_sin.astype(np.float64)
    hc = hour_cos.astype(np.float64)
    ds = doy_sin.astype(np.float64)
    dc = doy_cos.astype(np.float64)
    return np.column_stack(
        [
            g,
            g * hs,
            g * hc,
            g * ds,
            g * dc,
            g * z,
            g * hs * ds,
            g * hs * dc,
            g * hc * ds,
        ]
    )
