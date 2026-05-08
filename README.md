# itu-rs

`itu-rs` is a Rust implementation of selected ITU-R P-series atmospheric
propagation routines, ported from `python-itu-r` for fast Earth-space slant-path
attenuation calculations.

This first crate is intentionally scoped: it supports the propagation features
needed by the current Rust port, not the full `python-itu-r` API surface.

## Data Files

The working repository checkout includes the required ITU-R grids under
`itu-rs/data`, so local development is self-contained.

The crates.io package excludes those grids because the compressed data package is
larger than the crates.io `.crate` upload limit. For published-package use, set
`ITU_RS_DATA_DIR` to a directory containing the same `itur/data` layout copied
from `python-itu-r`:

```powershell
$env:ITU_RS_DATA_DIR = "C:\path\to\ITU-Rpy\itur\data"
```

Alternatively, enable the opt-in `data` feature to fetch and compile the grid
files into your build:

```toml
[dependencies]
itu-rs = { version = "1", features = ["data"] }
```

With `features = ["data"]`, the build script first uses local `data/` files when
they are present. Otherwise it downloads a source archive, extracts `data/**`,
and embeds those bytes into the compiled crate. The default download URL points
at this repository's GitHub archive. Override it with `ITU_RS_DATA_URL`, use a
local archive with `ITU_RS_DATA_ARCHIVE`, set `ITU_RS_DATA_CACHE` for a persistent
download cache, and set `ITU_RS_DATA_SHA256` to require checksum verification.

The feature is intentionally opt-in because it makes builds network-dependent
when local data is unavailable and increases the final binary size by roughly the
compressed grid data size.

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

| python-itu-r function or feature | Rust support | Status |
|---|---:|---|
| `itur.atmospheric_attenuation_slant_path` | ✅ | Public API |
| Batched slant-path attenuation over elevation angles | ✅ | Public API |
| Gas-only default slant-path attenuation helper | ✅ | Public API |
| P.676 gaseous attenuation, exact and approximate paths | ✅ | Internal/public through slant path |
| P.618 rain attenuation contribution | ✅ | Internal/public through contributions |
| P.618 scintillation attenuation contribution | ✅ | Internal/public through contributions |
| P.840 cloud attenuation contribution | ✅ | Internal/public through contributions |
| P.1511 topographic altitude lookup | ✅ | Internal data support |
| P.1510 surface mean temperature lookup | ✅ | Internal data support |
| P.836 water vapour density and total content lookup | ✅ | Internal data support |
| P.837 rainfall rate lookup | ✅ | Internal data support |
| P.839 rain height lookup | ✅ | Internal data support |
| P.453 wet-term radio refractivity lookup | ✅ | Internal data support |
| Full `python-itu-r` package API | ❌ | Out of scope for this crate |

## Long-Term Goals

The long-term goal is full ITU-R recommendation coverage, with Rust
implementations of the propagation models exposed through a stable native API.

Another goal is Python exposure as an installable Python package, likely through
PyO3 or a similar binding layer. That package should make it possible to compare
the Rust implementation directly against Python workflows and measure whether
the Rust core materially speeds up common `python-itu-r` use cases.

## Benchmarks

The current comparison benchmark was run from the source workspace on
May 8, 2026 using the existing Python parity harness against `python-itu-r` and
the Rust port. Results are means over repeated runs.

| Scenario | Python mean | Rust mean | Speedup | Max absolute error |
|---|---:|---:|---:|---:|
| Batched default slant path, 4 locations x 169 elevations | `0.195419 s` | `0.005600 s` | `34.89x` | `7.105e-15 dB` |
| Exact scalar slant path with explicit overrides | `0.140981 s` | `0.002637 s` | `53.46x` | `0.000e+00 dB` |

Run the Rust-only Criterion benchmarks with:

```powershell
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

`SlantPathOptions::default()` matches the default slant-path configuration used
by the port. Set `exact = true` to use exact gaseous attenuation.

## Validation

Current local checks:

```powershell
cargo check
cargo test
cargo test --doc
```

Before publishing, also run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo package --allow-dirty
```

The local source checkout packages to a small crates.io-compatible `.crate` by
excluding `data/**`. Keep the data directory in the repository for development
and parity testing, but do not include it in the crates.io upload unless the
registry size limit is handled another way.

## Attribution

This crate contains a Rust port of selected functionality from
[`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy), which is MIT
licensed. See `NOTICE.md`.
