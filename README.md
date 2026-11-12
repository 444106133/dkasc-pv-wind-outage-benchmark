# Photovoltaic Power Estimation Under Persistent Wind-Telemetry Loss

Generalized, site-configurable Python framework for multi-technology photovoltaic (PV) array reference power estimation, module thermal-chain decomposition, plane-of-array (POA) solar-geometry transposition, and fault-gated online conformal uncertainty calibration under persistent wind-sensor loss.

## Package Structure

- `pv_wind_benchmark/config.py`: Site coordinates, dataset column schema, temporal split definitions, model hyperparameters, thermal-chain constants, conformal calibration settings, and block-bootstrap configuration dataclasses.
- `pv_wind_benchmark/solar.py`: Deterministic solar-geometry calculations (Spencer equation of time, solar declination, hour angle, zenith cosine, cyclic clock-hour and day-of-year encodings, and nine-term harmonic POA transposition basis).
- `pv_wind_benchmark/data.py`: Multi-stream CSV/Parquet ingestion, service-window screening, physical-bounds filtering, temporal splitting, and per-stream training quantile scale computation.
- `pv_wind_benchmark/models.py`: Generalized estimator implementations:
  - `WindFreeBoostingEstimator`: Wind-free and wind-augmented histogram gradient boosting with optional post-fit annual degradation scaling.
  - `MeasuredWindFallbackHybrid`: Bounded physical linear core with hierarchical fleet/technology/stream polynomial ridge and bounded measured-wind residual correction.
  - `AvailabilityGatedMixtureOfExperts`: Irradiance and solar-zenith regime-partitioned simplex blend of physical, wind-free, and wind-aware experts with empirical Bayes count shrinkage.
  - `PhysicalThermalChainEstimator`: Faiman convective cooling and PVWatts temperature-derating chain supporting learned wind imputation, affine reanalysis wind calibration, technology-specific temperature coefficients, and harmonic POA transposition.
  - `WindValueDistillationGate`: Paired block-bootstrap value-of-information gate for wind-aware teacher models.
- `pv_wind_benchmark/conformal.py`: `SplitConformalCalibrator` and `OnlineConformalMonitor` implementing static symmetric/asymmetric, rolling, causal adaptive conformal inference (ACI), and causal yield-ratio fault-gated conformal intervals (`FG-Roll`, `FG-Asym`).
- `pv_wind_benchmark/evaluation.py`: Fleet micro MAE, median stream training-normalized and nameplate-normalized errors, signed bias, synchronized noncircular block bootstrap, and paired Wilcoxon signed-rank tests.
- `pv_wind_benchmark/cli.py`: Command-line runner executing the end-to-end training, calibration, and evaluation workflow from JSON configurations.
- `configs/`: Generalized site, hyperparameter, technology-code, and stream-metadata configurations (`benchmark_config.json`, `technology_codes.json`, `site_metadata.json`).

## Installation

```bash
pip install -e .
```

## Usage

```bash
pv-wind-benchmark \
  --data /path/to/prepared_records.parquet \
  --config configs/benchmark_config.json \
  --tech-codes configs/technology_codes.json \
  --metadata configs/site_metadata.json \
  --output-dir outputs
```

## License

MIT License. See `LICENSE` for details.
