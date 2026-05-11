# itu-rs

`itu-rs` is a Rust implementation of selected [ITU-R P-series](https://www.itu.int/rec/R-REC-p)
atmospheric propagation routines, ported from
[`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy) for fast Earth-space slant-path attenuation
calculations.

## Data Files

The working repository checkout includes the required [ITU-R](https://www.itu.int/pub/R-REC) grids under
`itu-rs/data`, so local development is self-contained.

For published-package use, the recommended path is to enable the `data` feature:

```toml
[dependencies]
itu-rs = { version = "1", features = ["data"] }
```

With `features = ["data"]`, no runtime configuration is required. The build
script first uses local `data/` files when they are present; otherwise it
downloads a source archive, extracts `data/**`, and embeds those bytes into the
compiled crate. The default download URL points at the `itu-rs-data-v1.zip`
GitHub Release asset and verifies its pinned SHA256.

The raw [ITU-R](https://www.itu.int/pub/R-REC) grids are too large to include directly in the crates.io package
because crates.io limits the size of uploaded `.crate` archives.

Override it with `ITU_RS_DATA_URL`, use a local archive with
`ITU_RS_DATA_ARCHIVE`, set `ITU_RS_DATA_CACHE` for a persistent download cache,
and set `ITU_RS_DATA_SHA256` to require a different checksum.

The feature is intentionally opt-in because it makes builds network-dependent
when local data is unavailable and increases the final binary size by roughly the
compressed grid data size.

### Manual data directory

If you do not want automatic data embedding, set `ITU_RS_DATA_DIR` to a
directory containing the same `itur/data` layout copied from
[`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy).

Unix shells:

```sh
export ITU_RS_DATA_DIR=/path/to/ITU-Rpy/itur/data
```

Windows PowerShell:

```powershell
$env:ITU_RS_DATA_DIR = "C:\path\to\ITU-Rpy\itur\data"
```

## Example

Compute the full atmospheric attenuation contribution set for one Earth-space
slant path:

```rust
use itu_rs::{atmospheric_attenuation_slant_path, SlantPathOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let attenuation = atmospheric_attenuation_slant_path(
        45.4215,   // latitude, degrees
        -75.6972,  // longitude, degrees
        12.0,      // frequency, GHz
        30.0,      // elevation, degrees
        0.1,       // time percentage
        1.2,       // antenna diameter, m
        SlantPathOptions::default(),
    )?;

    println!("{:.6} dB", attenuation.total_db);
    Ok(())
}
```

Sweep multiple elevation angles for one fixed site and link configuration:

```rust
use itu_rs::{atmospheric_attenuation_slant_path_many, SlantPathOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elevations = [5.0, 17.5, 45.0, 89.0];
    let attenuation = atmospheric_attenuation_slant_path_many(
        45.4215,
        -75.6972,
        12.0,
        &elevations,
        0.1,
        1.2,
        SlantPathOptions::default(),
    )?;

    for (elevation, result) in elevations.iter().zip(attenuation.iter()) {
        println!("{elevation:5.1} deg: {:.6} dB", result.total_db);
    }

    Ok(())
}
```

Use exact gaseous attenuation or disable individual components through
`SlantPathOptions`:

```rust
use itu_rs::{atmospheric_attenuation_slant_path, SlantPathOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = SlantPathOptions {
        exact: true,
        include_rain: false,
        include_clouds: false,
        ..SlantPathOptions::default()
    };

    let attenuation = atmospheric_attenuation_slant_path(
        10.0, 20.0, 18.0, 17.5, 0.7, 0.8, options,
    )?;

    println!("gas + scintillation: {:.6} dB", attenuation.total_db);
    Ok(())
}
```

## Supported Coverage

| [`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy) function or feature |  | Status |
|---|---:|---|
| `itur.atmospheric_attenuation_slant_path` | ✅ | Public API |
| Batched slant-path attenuation over elevation angles | ✅ | Public API |
| Gas-only default slant-path attenuation helper | ✅ | Public API |
| [P.676](https://www.itu.int/rec/R-REC-P.676) gaseous attenuation, exact and approximate paths | ✅ | Public scalar APIs |
| [P.618](https://www.itu.int/rec/R-REC-P.618) rain attenuation contribution | ✅ | Public scalar API |
| [P.618](https://www.itu.int/rec/R-REC-P.618) scintillation attenuation contribution | ✅ | Public scalar APIs |
| [P.840](https://www.itu.int/rec/R-REC-P.840) cloud attenuation contribution | ✅ | Public scalar APIs |
| [P.1511](https://www.itu.int/rec/R-REC-P.1511) topographic altitude lookup | ✅ | Public scalar API |
| [P.1510](https://www.itu.int/rec/R-REC-P.1510) surface mean temperature lookup | ✅ | Public scalar API |
| [P.836](https://www.itu.int/rec/R-REC-P.836) water vapour density and total content lookup | ✅ | Public scalar APIs |
| [P.837](https://www.itu.int/rec/R-REC-P.837) rainfall rate lookup | ✅ | Public scalar API |
| [P.839](https://www.itu.int/rec/R-REC-P.839) rain height lookup | ✅ | Public scalar API |
| [P.453](https://www.itu.int/rec/R-REC-P.453) wet-term radio refractivity lookup | ✅ | Public scalar APIs |
| [P.835](https://www.itu.int/rec/R-REC-P.835) reference atmosphere | ✅ | Public scalar APIs |
| [P.838](https://www.itu.int/rec/R-REC-P.838) rain specific attenuation | ✅ | Public scalar APIs |
| [P.618](https://www.itu.int/rec/R-REC-P.618) rain attenuation probability | ❌ | Not implemented |
| [P.618](https://www.itu.int/rec/R-REC-P.618) site-diversity rain outage probability | ❌ | Not implemented |
| [P.530](https://www.itu.int/rec/R-REC-P.530) terrestrial line-of-sight paths | ❌ | Not implemented |
| [P.1144](https://www.itu.int/rec/R-REC-P.1144) interpolation helper APIs | ❌ | Not exposed as public APIs |
| [P.1623](https://www.itu.int/rec/R-REC-P.1623) fade duration, fade slope, and fade depth | ❌ | Not implemented |
| [P.1853](https://www.itu.int/rec/R-REC-P.1853) tropospheric impairment time-series synthesis | ❌ | Not implemented |

## Benchmarks

The current comparison benchmark was run from the source workspace on
May 8, 2026 using the existing Python parity harness against
[`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy) and
the Rust port. Results are means over repeated runs.

| Scenario | Python mean | Rust mean | Speedup | Max absolute error |
|---|---:|---:|---:|---:|
| Batched default slant path, 4 locations x 169 elevations | `0.195419 s` | `0.005600 s` | `34.89x` | `7.105e-15 dB` |
| Exact scalar slant path with explicit overrides | `0.140981 s` | `0.002637 s` | `53.46x` | `0.000e+00 dB` |

Run the Rust-only Criterion benchmarks with:

```sh
cargo bench
```

## API

The primary public calls are:

| Function | Purpose |
|---|---|
| `atmospheric_attenuation_slant_path` | Compute gas, cloud, rain, scintillation, and total attenuation for one elevation angle. |
| `atmospheric_attenuation_slant_path_many` | Compute the same contribution set for many elevation angles. |
| `gas_attenuation_default` | Compute gas-only default attenuation for one elevation angle. |
| `gas_attenuation_default_many` | Compute gas-only default attenuation for many elevation angles. |
| `gas_attenuation_default_many_checked` | Compatibility alias for strict gas-only batch validation. |

Additional scalar APIs expose the implemented recommendation pieces directly:

| Recommendation | Public functions |
|---|---|
| [P.1510](https://www.itu.int/rec/R-REC-P.1510)/[P.1511](https://www.itu.int/rec/R-REC-P.1511) | `surface_mean_temperature_k`, `topographic_altitude_km` |
| [P.835](https://www.itu.int/rec/R-REC-P.835) | `standard_temperature_k`, `standard_pressure_hpa`, `standard_water_vapour_density_gm3` |
| [P.836](https://www.itu.int/rec/R-REC-P.836)/[P.837](https://www.itu.int/rec/R-REC-P.837)/[P.839](https://www.itu.int/rec/R-REC-P.839) | `surface_water_vapour_density_gm3`, `total_water_vapour_content_kgm2`, `rainfall_rate_r001_mmh`, `rain_height_km` |
| [P.838](https://www.itu.int/rec/R-REC-P.838) | `rain_specific_attenuation_coefficients`, `rain_specific_attenuation_db_per_km` |
| [P.840](https://www.itu.int/rec/R-REC-P.840) | `cloud_reduced_liquid_kgm2`, `cloud_liquid_mass_absorption_coefficient`, `cloud_specific_attenuation_coefficient`, `cloud_attenuation_db` |
| [P.453](https://www.itu.int/rec/R-REC-P.453) | `wet_term_radio_refractivity`, `radio_refractive_index`, `water_vapour_pressure_hpa`, `map_wet_term_radio_refractivity` |
| [P.676](https://www.itu.int/rec/R-REC-P.676) | `gamma0_exact_db_per_km`, `gammaw_exact_db_per_km`, `gamma_exact_db_per_km`, `slant_inclined_path_equivalent_height_km`, `zenith_water_vapour_attenuation_db`, `gaseous_attenuation_slant_path_db` |
| [P.618](https://www.itu.int/rec/R-REC-P.618) | `rain_attenuation_db`, `scintillation_sigma_db`, `scintillation_attenuation_db` |

`SlantPathOptions::default()` matches the default slant-path configuration used
by the port. Set `exact = true` to use exact gaseous attenuation.

## Attribution

This crate contains a Rust port of selected functionality from
[`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy), which is MIT licensed. See `NOTICE.md`.

