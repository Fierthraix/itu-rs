# Changelog

All notable user-facing changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a Python package, published as `itu-rs` and imported as `itu_rs`, backed
  by the same Rust implementation.
- Added Python bindings for the public Rust APIs, including slant-path,
  gas-only, scalar recommendation, and atmospheric helper functions.
- Added Python classes for `SlantPathOptions`, `SlantPathContributions`, and
  `HydrometeorType`.
- Added Python wheels that embed the ITU-R data grids, so Python users do not
  need `ITU_RS_DATA_DIR` for normal installs.

## [1.2.0] - 2026-05-11

### Added

- Added `HydrometeorType` for selecting water or ice saturation-vapour pressure calculations.
- Added monthly P.1510 temperature lookup through `surface_month_mean_temperature_k`.
- Added P.837 rain-occurrence and arbitrary-percent rain-rate APIs:
  `rainfall_probability_percent`, `rainfall_rate_mmh`, and
  `unavailability_from_rainfall_rate_percent`.
- Added `zero_isotherm_height_km` for direct P.839 zero-degree isotherm lookup.
- Added P.840 lognormal cloud APIs:
  `lognormal_approximation_coefficients` and
  `cloud_attenuation_lognormal_db`.
- Added P.453 refractivity-gradient map APIs: `dn65` and `dn1`.
- Added P.453/P.835 atmosphere helper APIs:
  `dry_term_radio_refractivity` and `saturation_vapour_pressure_hpa`.
- Added P.678 inter-annual variability APIs:
  `inter_annual_variability` and `risk_of_exceedance`.
- Added the ITU-R data grids required by the new P.1510, P.453, P.678,
  P.837, and P.840 APIs.

### Changed

- Updated the default bundled data archive to `itu-rs-data-v2` for users of
  the `data` feature.
- Updated README API coverage and benchmark documentation for the expanded
  public API.

## [1.1.5] - 2026-05-11

### Changed

- Added README badges for CI, crates.io, downloads, docs.rs, and license status.

## [1.1.4] - 2026-05-11

### Changed

- Improved compatibility of the optional bundled-data workflow by updating its
  build-time dependencies.

## [1.1.3] - 2026-05-11

### Changed

- Clarified data-file documentation for repository checkouts, published crate
  usage, and manual `ITU_RS_DATA_DIR` configuration.

## [1.1.2] - 2026-05-11

### Changed

- Refreshed crate metadata and README links for crates.io and docs.rs users.
- Expanded README examples and supported-coverage documentation.

## [1.1.1] - 2026-05-10

### Changed

- Expanded public API documentation with parameters, return values, errors, and
  examples.

## [1.1.0] - 2026-05-08

### Added

- Added direct public scalar APIs for implemented ITU-R recommendation pieces,
  including P.1511, P.1510, P.835, P.836, P.837, P.839, P.838, P.840,
  P.453, P.676, and P.618.
- Added README coverage tables for the supported recommendation APIs.

## [1.0.2] - 2026-05-08

### Fixed

- Pinned the bundled data feature to a specific release asset and checksum for
  reproducible builds.

## [1.0.1] - 2026-05-08

### Added

- Added the optional `data` feature, which downloads and embeds the required
  ITU-R data archive at build time when local data files are not present.

### Changed

- Updated the crate to Rust 2024 edition.
- Updated README, examples, and benchmarks for the optional bundled-data
  workflow.

## [1.0.0] - 2026-05-08

### Added

- Added the first stable Rust API for ITU-R Earth-space slant-path attenuation.
- Added `atmospheric_attenuation_slant_path` and
  `atmospheric_attenuation_slant_path_many` for gas, cloud, rain,
  scintillation, and total attenuation.
- Added gas-only convenience APIs:
  `gas_attenuation_default`, `gas_attenuation_default_many`, and
  `gas_attenuation_default_many_checked`.
- Added `SlantPathOptions`, `SlantPathContributions`, and `ItuError`.
- Added repository data files needed for local self-contained development.
- Added an example program and Criterion benchmark for slant-path attenuation.

## [0.1.0] - 2026-05-08

### Added

- Started the crate with package metadata, licensing, attribution, README
  documentation, and the initial implementation work that led to `1.0.0`.
