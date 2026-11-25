# DKASC PV Power Estimation and Fault-Gated Conformal Monitoring

Reference code and site configurations for *"Photovoltaic Power Estimation Under Persistent Wind-Telemetry Loss Across a Decade-Scale Multi-Technology Fleet"* (A. M. Bin Dayil, E. A. Al-Ammar, Y. Bin Salamah, S. Almohaimeed, M. R. Aboelmagd, and M. A. Al-Haj, Department of Electrical Engineering, King Saud University).

At the Desert Knowledge Australia Solar Centre (DKASC) in Alice Springs (`23.76 deg S, 133.88 deg E`), the station anemometer failed permanently on October 21, 2016. All models here are fitted once on 2013-2015 five-minute daytime telemetry (`G_tilt >= 50 W/m^2`) and evaluated out of sample through January 20, 2026 across 36 scored arrays spanning seven PV technologies (29 fixed-tilt streams and 7 single- or dual-axis silicon and CPV trackers).

Two reference model families isolate wind availability from plane-of-array (POA) solar geometry: Faiman-PVWatts thermal chains (`Phys-Faiman` on shared `G_tilt` and `Phys-POA` with harmonic solar-geometry transposition) and a single histogram gradient-boosted tree architecture (`HGB-WF` without wind, compared against `HGB-Wind`, `HGB-Imputed`, and affine-calibrated 10-m ERA5 reanalysis `HGB-ERA5`). For online uncertainty tracking across 21 provider-logged O&M fault windows over 2016-2026 (2 pre-outage 2016 major faults, `F07` and `F18`, and 19 post-outage 2017-2026 windows: 18 major plus partial string outage `F21`), `OnlineConformalMonitor` implements static split conformal intervals, unguarded rolling and causal adaptive conformal updates (`Rolling-30d`, `Causal ACI`, `Asym-Roll-30d`), and causal trailing 7-day yield-ratio fault-gated asymmetric intervals (`FG-Asym-30d`).

## Repository Layout

```text
pv_wind_benchmark/
  config.py       Site bounds, split timestamps, tree, thermal, conformal, and bootstrap configs
  solar.py        Spencer solar geometry and nine-term harmonic POA basis
  data.py         CSV/Parquet loading, service-window filtering, and training scale estimation
  models.py       Phys-Faiman / Phys-POA thermal chains and HGB reference estimators
  conformal.py    Split conformal calibration and fault-gated online conformal monitoring
  evaluation.py   STC nameplate metrics (nMAE, nRMSE, nMBE, R^2), bias decomposition, and block bootstrap
  cli.py          End-to-end training, evaluation, and sensitivity sweep runner
configs/
  benchmark_config.json   Frozen 2013-2015 splits and model hyperparameters
  site_metadata.json      40 DKASC Alice Springs stream definitions and 21 O&M fault windows (F01-F21)
  technology_codes.json   Integer technology class mapping
```

## Data Sources and Input Format

Public five-minute array and weather exports are openly available from the [DKASC Alice Springs portal](https://dkasolarcentre.com.au/download?location=alice-springs), and hourly 10-m reanalysis wind (`reanalysis_wind_ms`) is available from the [ECMWF ERA5 single-levels archive](https://doi.org/10.24381/cds.adbb2d47). Combine the site CSVs into a single table (`.parquet` or `.csv`) with the columns configured in `configs/benchmark_config.json`:

| Column | Units | Notes |
|---|---|---|
| `timestamp` | Local ISO-8601 (`UTC+9.5`) | Five-minute interval timestamp |
| `stream_id` | Integer | DKASC source ID matching `configs/site_metadata.json` |
| `technology` | String or integer code | Absorber class (`poly-Si`, `mono-Si`, `CdTe`, `CIGS`, `HIT`, `a-Si`, `CPV`) |
| `active_power` | `kW` | Measured AC active power |
| `poa_irradiance` | `W/m^2` | Shared `20 deg` North tilted global pyranometer (`G_tilt`) |
| `ambient_temp_c` | `deg C` | Station ambient air temperature (`T_a`) |
| `wind_speed_ms` | `m/s` | Station wind speed (valid prior to `2016-10-21T13:05:00`, `NaN` thereafter) |
| `reanalysis_wind_ms` | `m/s` | Optional interpolated 10-m ERA5 wind speed |

## Installation and Execution

Requires Python 3.11 or newer.

```bash
pip install -e .

pv-wind-benchmark \
  --data /path/to/dkasc_prepared.parquet \
  --config configs/benchmark_config.json \
  --tech-codes configs/technology_codes.json \
  --metadata configs/site_metadata.json \
  --output-dir outputs
```

Running the CLI writes `outputs/benchmark_summary.json` with pre-outage 2016 and wind-free 2017-2026 goodness-of-fit tables (`fixed_tilt` and `all_streams`), 7-day block-bootstrap confidence intervals, paired stream Wilcoxon signed-rank `p`-values, post-2015 `nMBE` cohort shares, O&M-log conformal interval tables, and the `(tau_rho, W_d)` sensitivity grid.

## License

MIT License (`LICENSE`).
