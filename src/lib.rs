//! Selected [ITU-R P-series](https://www.itu.int/rec/R-REC-p) atmospheric propagation models.
//!
//! This crate is a Rust port of the subset of
//! [`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy) needed for
//! Earth-space atmospheric attenuation on a slant path. It currently focuses on
//! `atmospheric_attenuation_slant_path` and the recommendation data needed to
//! compute gas, cloud, rain, scintillation, and total attenuation contributions.
//!
//! # Data files
//!
//! Model grids are loaded lazily on first use. In a repository checkout, the
//! crate looks for data under `data/` next to `Cargo.toml`. For published
//! package use, enable `features = ["data"]` so the build downloads, verifies,
//! and embeds the required data automatically with no runtime configuration.
//!
//! The raw [ITU-R](https://www.itu.int/pub/R-REC) grids are too large to
//! include directly in the crates.io package because crates.io limits the size
//! of uploaded `.crate` archives.
//!
//! If automatic data embedding is not suitable, set `ITU_RS_DATA_DIR` to a
//! directory containing the
//! [`python-itu-r`](https://github.com/inigodelportillo/ITU-Rpy) `itur/data`
//! layout. On Unix shells:
//!
//! ```sh
//! export ITU_RS_DATA_DIR=/path/to/ITU-Rpy/itur/data
//! ```
//!
//! On Windows PowerShell:
//!
//! ```powershell
//! $env:ITU_RS_DATA_DIR = "C:\path\to\ITU-Rpy\itur\data"
//! ```
//!
//! The examples that depend on recommendation grids check whether data is
//! available before calling the model. That keeps doctests useful in this
//! repository, with `features = ["data"]`, and in downstream builds that set
//! `ITU_RS_DATA_DIR`.
//!
//! # Conventions
//!
//! Public APIs use plain `f64` values and encode units in argument and return
//! names:
//!
//! - `_deg`: degrees. `lat_deg` is geodetic latitude and must be in
//!   `[-90, 90]`. `lon_deg` is geodetic longitude; any finite degree value is
//!   accepted and map lookups wrap it internally.
//! - `_ghz`: carrier frequency in gigahertz.
//! - `_km` and `_m`: kilometres and metres. Site heights are above mean sea
//!   level unless the function says otherwise.
//! - `_hpa`: pressure in hectopascals.
//! - `_gm3`: density in grams per cubic metre.
//! - `_kgm2`: columnar mass in kilograms per square metre.
//! - `_mmh`: rainfall rate in millimetres per hour.
//! - `_db` and `_db_per_km`: attenuation in decibels and specific attenuation
//!   in decibels per kilometre.
//! - `_k` and `_c`: temperature in kelvin and degrees Celsius.
//!
//! The `p` argument is a percentage of time exceeded, not a probability
//! fraction. For example, `0.1` means 0.1% of an average year. Elevation angles
//! are above the local horizon and must be in the open interval `(0, 90)`.
//!
//! # Example
//!
//! ```
//! # fn data_available() -> bool {
//! #     cfg!(feature = "data")
//! #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
//! #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
//! #             .join("data/1511/v2_lat.npz")
//! #             .exists()
//! # }
//! # if data_available() {
//! use itu_rs::{atmospheric_attenuation_slant_path, SlantPathOptions};
//!
//! let attenuation = atmospheric_attenuation_slant_path(
//!     45.4215,   // latitude, degrees
//!     -75.6972,  // longitude, degrees
//!     12.0,      // frequency, GHz
//!     30.0,      // elevation, degrees
//!     0.1,       // time percentage exceeded
//!     1.2,       // antenna diameter, m
//!     SlantPathOptions::default(),
//! )?;
//!
//! assert!(attenuation.total_db.is_finite());
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!

use ndarray::{Array2, Axis};
use ndarray_npy::NpzReader;
use std::borrow::Cow;
use std::f64::consts::FRAC_PI_2;
use std::io::{BufRead, BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[cfg(feature = "data")]
mod bundled_data {
    include!(concat!(env!("OUT_DIR"), "/bundled_data.rs"));
}

#[cfg(not(feature = "data"))]
mod bundled_data {
    pub fn get(_rel_path: &str) -> Option<&'static [u8]> {
        None
    }
}

/// Error returned by ITU-R model loading, validation, and calculation routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItuError {
    message: String,
}

impl ItuError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the human-readable error message.
    ///
    /// The message names the input or model-loading problem that caused the
    /// operation to fail. Validation errors do not require ITU-R data to be
    /// loaded, so they are useful for checking bad inputs before any grid files
    /// are touched.
    ///
    /// # Example
    ///
    /// ```
    /// let err = itu_rs::rainfall_rate_r001_mmh(91.0, 0.0).unwrap_err();
    ///
    /// assert_eq!(err.message(), "lat_deg must be in [-90, 90]");
    /// ```
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for ItuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ItuError {}

impl From<String> for ItuError {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

const EPSILON: f64 = 1e-9;

const P836_LEVELS: [f64; 18] = [
    0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0,
    99.0,
];

const P840_LEVELS: [f64; 23] = [
    0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0,
    70.0, 80.0, 90.0, 95.0, 99.0, 100.0,
];

const P453_LEVELS: [f64; 18] = [
    0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0,
    99.0,
];

const HW_COEFFS_V13: [(f64, f64, f64); 3] = [
    (22.235080, 2.6846, 2.7649),
    (183.310087, 5.8905, 4.9219),
    (325.152888, 2.9810, 3.0748),
];

const HW_A_V13: f64 = 5.6585e-5;
const HW_B_V13: f64 = 1.8348;
const EXACT_GAS_LAYERS: usize = 922;

static MODEL: OnceLock<Result<IturModel, String>> = OnceLock::new();

#[derive(Clone)]
struct SpectralLine {
    f: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    c5: f64,
    c6: f64,
}

#[derive(Clone)]
struct H0Coefficients {
    freq_ghz: f64,
    a0: f64,
    b0: f64,
    c0: f64,
    d0: f64,
}

struct RegularGrid2D {
    lat_axis: Vec<f64>,
    lon_axis: Vec<f64>,
    values: Array2<f64>,
}

impl RegularGrid2D {
    fn from_npz(
        rel_lat: &str,
        rel_lon: &str,
        rel_values: &str,
        flip_ud: bool,
    ) -> Result<Self, String> {
        let lat_grid = load_npz_array2(rel_lat)?;
        let lon_grid = load_npz_array2(rel_lon)?;
        let mut values = load_npz_array2(rel_values)?;

        if lat_grid.nrows() == 0 || lat_grid.ncols() == 0 {
            return Err(format!("grid is empty for {rel_values}"));
        }

        let mut lat_axis: Vec<f64> = lat_grid.column(0).iter().copied().collect();
        let mut lon_axis: Vec<f64> = lon_grid.row(0).iter().copied().collect();

        if flip_ud {
            lat_axis.reverse();
            values.invert_axis(Axis(0));
        }

        if lat_axis.first().copied().unwrap_or(0.0) > lat_axis.last().copied().unwrap_or(0.0) {
            lat_axis.reverse();
            values.invert_axis(Axis(0));
        }

        if lon_axis.first().copied().unwrap_or(0.0) > lon_axis.last().copied().unwrap_or(0.0) {
            lon_axis.reverse();
            values.invert_axis(Axis(1));
        }

        Ok(Self {
            lat_axis,
            lon_axis,
            values,
        })
    }

    fn interp(&self, lat: f64, lon: f64) -> f64 {
        if !lat.is_finite() || !lon.is_finite() {
            return f64::NAN;
        }

        let (lat_lo, lat_hi, lat_frac) = bracket(&self.lat_axis, lat);
        let (lon_lo, lon_hi, lon_frac) = bracket(&self.lon_axis, lon);

        let v00 = self.values[[lat_lo, lon_lo]];
        let v10 = self.values[[lat_hi, lon_lo]];
        let v01 = self.values[[lat_lo, lon_hi]];
        let v11 = self.values[[lat_hi, lon_hi]];

        let v0 = v00 * (1.0 - lat_frac) + v10 * lat_frac;
        let v1 = v01 * (1.0 - lat_frac) + v11 * lat_frac;
        v0 * (1.0 - lon_frac) + v1 * lon_frac
    }
}

struct BicubicGrid2D {
    lat_axis: Vec<f64>,
    lon_axis: Vec<f64>,
    values: Array2<f64>,
}

impl BicubicGrid2D {
    fn from_npz(
        rel_lat: &str,
        rel_lon: &str,
        rel_values: &str,
        flip_ud: bool,
    ) -> Result<Self, String> {
        let lat_grid = load_npz_array2(rel_lat)?;
        let lon_grid = load_npz_array2(rel_lon)?;
        let mut values = load_npz_array2(rel_values)?;

        if lat_grid.nrows() < 4 || lat_grid.ncols() < 4 {
            return Err(format!("bicubic grid is too small for {rel_values}"));
        }

        let mut lat_axis: Vec<f64> = lat_grid.column(0).iter().copied().collect();
        let mut lon_axis: Vec<f64> = lon_grid.row(0).iter().copied().collect();

        if flip_ud
            || lat_axis.first().copied().unwrap_or(0.0) > lat_axis.last().copied().unwrap_or(0.0)
        {
            lat_axis.reverse();
            values.invert_axis(Axis(0));
        }

        if lon_axis.first().copied().unwrap_or(0.0) > lon_axis.last().copied().unwrap_or(0.0) {
            lon_axis.reverse();
            values.invert_axis(Axis(1));
        }

        Ok(Self {
            lat_axis,
            lon_axis,
            values,
        })
    }

    fn interp(&self, lat: f64, lon: f64) -> f64 {
        if !lat.is_finite() || !lon.is_finite() {
            return f64::NAN;
        }

        let lat_row = &self.lat_axis[1..self.lat_axis.len() - 1];
        let lon_row = &self.lon_axis[1..self.lon_axis.len() - 1];
        let lat_step = lat_row[1] - lat_row[0];
        let lon_step = lon_row[1] - lon_row[0];

        let mut r_idx = ((searchsorted_right(lat_row, lat) as isize - 1)
            + (searchsorted_left(lat_row, lat) as isize - 1))
            / 2;
        let mut c_idx = ((searchsorted_right(lon_row, lon) as isize - 1)
            + (searchsorted_right(lon_row, lon) as isize - 1))
            / 2;

        r_idx = r_idx.clamp(0, self.values.nrows() as isize - 4);
        c_idx = c_idx.clamp(0, self.values.ncols() as isize - 4);

        let r = (lat - lat_row[0]) / lat_step + 1.0;
        let c = (lon - lon_row[0]) / lon_step + 1.0;

        let r0 = r_idx as usize;
        let c0 = c_idx as usize;

        let mut row_accum = [0.0_f64; 4];
        for (dr, row_value) in row_accum.iter_mut().enumerate() {
            let rr = r0 + dr;
            *row_value = self.values[[rr, c0]] * kernel(c - c0 as f64)
                + self.values[[rr, c0 + 1]] * kernel(c - (c0 + 1) as f64)
                + self.values[[rr, c0 + 2]] * kernel(c - (c0 + 2) as f64)
                + self.values[[rr, c0 + 3]] * kernel(c - (c0 + 3) as f64);
        }

        row_accum[0] * kernel(r - r0 as f64)
            + row_accum[1] * kernel(r - (r0 + 1) as f64)
            + row_accum[2] * kernel(r - (r0 + 2) as f64)
            + row_accum[3] * kernel(r - (r0 + 3) as f64)
    }
}

struct IturModel {
    topo_1511_v2: BicubicGrid2D,
    temp_1510_v1: RegularGrid2D,
    topo_836_v6: BicubicGrid2D,
    rho_836_v6: Vec<(f64, RegularGrid2D)>,
    v_836_v6: Vec<(f64, RegularGrid2D)>,
    vsch_836_v6: Vec<(f64, RegularGrid2D)>,
    oxygen_lines_v13: Vec<SpectralLine>,
    water_lines_v13: Vec<SpectralLine>,
    h0_coeffs_v13: Vec<H0Coefficients>,
    rainfall_r001_837_v7: RegularGrid2D,
    zero_isotherm_839_v4: RegularGrid2D,
    cloud_lred_840_v9: Vec<(f64, RegularGrid2D)>,
    wet_refractivity_453_v13: Vec<(f64, RegularGrid2D)>,
}

#[derive(Clone, Copy, Debug)]
/// Optional environmental and model inputs for slant-path attenuation.
///
/// Defaults match the ported `python-itu-r` slant-path behavior: all
/// environmental inputs are looked up from the bundled
/// [ITU-R](https://www.itu.int/pub/R-REC) grids where possible, approximate
/// gaseous attenuation is used, and gas, cloud, rain, and scintillation
/// contributions are included.
pub struct SlantPathOptions {
    /// Earth station altitude above mean sea level in kilometres.
    ///
    /// This is the ground terminal height used by rain, gas, and water-vapour
    /// calculations. It may be negative for sites below sea level. When `None`,
    /// the model uses the [P.1511](https://www.itu.int/rec/R-REC-P.1511)/[P.836](https://www.itu.int/rec/R-REC-P.836) topographic map at the requested site.
    pub hs_km: Option<f64>,
    /// Surface water vapour density in g/m^3.
    ///
    /// This is the near-surface absolute humidity used by gaseous attenuation.
    /// Values must be non-negative. When `None`, [P.836](https://www.itu.int/rec/R-REC-P.836) water-vapour maps are
    /// used for the site and time percentage.
    pub rho_gm3: Option<f64>,
    /// Rainfall rate exceeded for 0.01% of an average year, in mm/h.
    ///
    /// This is the intense-rain reference rate used by [P.618](https://www.itu.int/rec/R-REC-P.618) to scale rain
    /// attenuation for the requested time percentage. Values must be
    /// non-negative. When `None`, [P.837](https://www.itu.int/rec/R-REC-P.837) data are used.
    pub r001_mmh: Option<f64>,
    /// Antenna efficiency factor used in scintillation attenuation.
    ///
    /// This dimensionless factor describes effective aperture efficiency for
    /// the scintillation fade calculation. It must be in `(0, 1]`; the default
    /// is `0.5`.
    pub eta: f64,
    /// Surface temperature in kelvin for gaseous attenuation.
    ///
    /// This temperature is used by gaseous attenuation and, when
    /// `h_percent` is provided, converted to Celsius for scintillation humidity
    /// calculations. Values must be greater than zero. When `None`, [P.1510](https://www.itu.int/rec/R-REC-P.1510)
    /// annual mean surface temperature data are used.
    pub t: Option<f64>,
    /// Relative humidity percentage for scintillation attenuation.
    ///
    /// Values must be in `[0, 100]`. When `None`, scintillation uses the
    /// recommendation's wet-term refractivity map instead of local humidity,
    /// temperature, and pressure.
    pub h_percent: Option<f64>,
    /// Atmospheric pressure in hPa.
    ///
    /// This is the surface pressure used by gaseous attenuation and, when
    /// `h_percent` is provided, scintillation water-vapour pressure. Values
    /// must be positive. When `None`, [P.835](https://www.itu.int/rec/R-REC-P.835) standard-atmosphere pressure is
    /// computed from `hs_km`.
    pub pressure_hpa: Option<f64>,
    /// Turbulent layer height in metres for scintillation attenuation.
    ///
    /// This is the effective height of the tropospheric turbulent layer. It
    /// must be positive; the default is `1000.0` m.
    pub h_l_m: f64,
    /// Slant-path length through rain in kilometres.
    ///
    /// This is the portion of the Earth-space path inside the rain region.
    /// Values must be positive when provided. When `None`, [P.618](https://www.itu.int/rec/R-REC-P.618) derives the
    /// path length from rain height, station height, and elevation.
    pub l_s_km: Option<f64>,
    /// Polarization tilt angle in degrees.
    ///
    /// This describes the radio wave polarization relative to horizontal and
    /// affects [P.838](https://www.itu.int/rec/R-REC-P.838) rain specific attenuation coefficients. It must be in
    /// `[0, 90]`; the default is `45.0`.
    pub tau_deg: f64,
    /// Total water vapour content in kg/m^2 for gaseous attenuation.
    ///
    /// This is the vertically integrated water-vapour column above the site.
    /// Values must be non-negative. When `None`, [P.836](https://www.itu.int/rec/R-REC-P.836) total-content maps are
    /// used.
    pub v_t_kgm2: Option<f64>,
    /// Use exact gaseous attenuation when `true`; use the faster approximate
    /// path when `false`.
    ///
    /// Exact mode integrates the [P.676](https://www.itu.int/rec/R-REC-P.676) gaseous model through atmosphere layers.
    /// Approximate mode is faster and matches the default slant-path behavior
    /// of the port. The default is `false`.
    pub exact: bool,
    /// Include the rain attenuation contribution in the returned total.
    ///
    /// When `false`, `SlantPathContributions::rain_db` is `0.0`.
    pub include_rain: bool,
    /// Include the gaseous attenuation contribution in the returned total.
    ///
    /// When `false`, `SlantPathContributions::gas_db` is `0.0`.
    pub include_gas: bool,
    /// Include the scintillation attenuation contribution in the returned total.
    ///
    /// When `false`, `SlantPathContributions::scintillation_db` is `0.0`.
    pub include_scintillation: bool,
    /// Include the cloud attenuation contribution in the returned total.
    ///
    /// When `false`, `SlantPathContributions::cloud_db` is `0.0`.
    pub include_clouds: bool,
}

#[derive(Clone, Copy, Debug)]
/// Atmospheric attenuation contribution breakdown in dB.
///
/// The fields separate the mechanisms used by the supported Earth-space
/// slant-path model. `total_db` is not a plain sum of all fields; it follows the
/// ported combination rule:
///
/// `gas_db + sqrt((rain_db + cloud_db)^2 + scintillation_db^2)`.
pub struct SlantPathContributions {
    /// Gaseous absorption contribution in dB.
    ///
    /// This comes from oxygen and water-vapour absorption along the slant path.
    pub gas_db: f64,
    /// Cloud liquid-water attenuation contribution in dB.
    ///
    /// This comes from reduced liquid water content and cloud mass absorption.
    pub cloud_db: f64,
    /// Rain fade contribution in dB.
    ///
    /// This comes from the [P.618](https://www.itu.int/rec/R-REC-P.618) rain model using rain rate, rain height,
    /// elevation, path geometry, and polarization.
    pub rain_db: f64,
    /// Tropospheric scintillation contribution in dB.
    ///
    /// This represents short-term amplitude fluctuations caused by turbulence.
    pub scintillation_db: f64,
    /// Total attenuation in dB.
    ///
    /// This follows the same combination rule as `python-itu-r` for the
    /// supported slant-path model:
    /// `gas_db + sqrt((rain_db + cloud_db)^2 + scintillation_db^2)`.
    pub total_db: f64,
}

impl Default for SlantPathOptions {
    fn default() -> Self {
        Self {
            hs_km: None,
            rho_gm3: None,
            r001_mmh: None,
            eta: 0.5,
            t: None,
            h_percent: None,
            pressure_hpa: None,
            h_l_m: 1000.0,
            l_s_km: None,
            tau_deg: 45.0,
            v_t_kgm2: None,
            exact: false,
            include_rain: true,
            include_gas: true,
            include_scintillation: true,
            include_clouds: true,
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl IturModel {
    fn load() -> Result<Self, String> {
        let topo_1511_v2 = BicubicGrid2D::from_npz(
            "1511/v2_lat.npz",
            "1511/v2_lon.npz",
            "1511/v2_topo.npz",
            true,
        )?;
        let temp_1510_v1 = RegularGrid2D::from_npz(
            "1510/v1_lat.npz",
            "1510/v1_lon.npz",
            "1510/v1_t_annual.npz",
            true,
        )?;
        let topo_836_v6 = BicubicGrid2D::from_npz(
            "836/v6_topolat.npz",
            "836/v6_topolon.npz",
            "836/v6_topo_0dot5.npz",
            true,
        )?;

        let mut rho_836_v6 = Vec::with_capacity(P836_LEVELS.len());
        let mut v_836_v6 = Vec::with_capacity(P836_LEVELS.len());
        let mut vsch_836_v6 = Vec::with_capacity(P836_LEVELS.len());
        for p in P836_LEVELS {
            let suffix = p836_suffix(p);
            rho_836_v6.push((
                p,
                RegularGrid2D::from_npz(
                    "836/v6_lat.npz",
                    "836/v6_lon.npz",
                    &format!("836/v6_rho_{suffix}.npz"),
                    false,
                )?,
            ));
            v_836_v6.push((
                p,
                RegularGrid2D::from_npz(
                    "836/v6_lat.npz",
                    "836/v6_lon.npz",
                    &format!("836/v6_v_{suffix}.npz"),
                    false,
                )?,
            ));
            vsch_836_v6.push((
                p,
                RegularGrid2D::from_npz(
                    "836/v6_lat.npz",
                    "836/v6_lon.npz",
                    &format!("836/v6_vsch_{suffix}.npz"),
                    false,
                )?,
            ));
        }

        let oxygen_lines_v13 = load_spectral_lines("676/v13_lines_oxygen.txt")?;
        let water_lines_v13 = load_spectral_lines("676/v13_lines_water_vapour.txt")?;
        let h0_coeffs_v13 = load_h0_coefficients("676/v13_h0_coefficients.txt")?;
        let rainfall_r001_837_v7 = RegularGrid2D::from_npz(
            "837/v7_lat_r001.npz",
            "837/v7_lon_r001.npz",
            "837/v7_r001.npz",
            true,
        )?;
        let zero_isotherm_839_v4 = RegularGrid2D::from_npz(
            "839/v4_esalat.npz",
            "839/v4_esalon.npz",
            "839/v4_esa0height.npz",
            false,
        )?;

        let mut cloud_lred_840_v9 = Vec::with_capacity(P840_LEVELS.len());
        for p in P840_LEVELS {
            let stem = p840_stem(p)?;
            cloud_lred_840_v9.push((
                p,
                RegularGrid2D::from_npz(
                    "840/v9_lat.npz",
                    "840/v9_lon.npz",
                    &format!("840/v9_l_{stem}.npz"),
                    false,
                )?,
            ));
        }

        let mut wet_refractivity_453_v13 = Vec::with_capacity(P453_LEVELS.len());
        for p in P453_LEVELS {
            let suffix = p453_suffix(p);
            wet_refractivity_453_v13.push((
                p,
                RegularGrid2D::from_npz(
                    "453/v13_lat_n.npz",
                    "453/v13_lon_n.npz",
                    &format!("453/v13_nwet_annual_{suffix}.npz"),
                    true,
                )?,
            ));
        }

        Ok(Self {
            topo_1511_v2,
            temp_1510_v1,
            topo_836_v6,
            rho_836_v6,
            v_836_v6,
            vsch_836_v6,
            oxygen_lines_v13,
            water_lines_v13,
            h0_coeffs_v13,
            rainfall_r001_837_v7,
            zero_isotherm_839_v4,
            cloud_lred_840_v9,
            wet_refractivity_453_v13,
        })
    }

    fn topographic_altitude_km(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        let lon_180 = wrap_lon_180(lon_deg);
        (self.topo_1511_v2.interp(lat_deg, lon_180) / 1000.0).max(EPSILON)
    }

    fn surface_mean_temperature_k(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        let lon_180 = wrap_lon_180(lon_deg);
        self.temp_1510_v1.interp(lat_deg, lon_180)
    }

    fn standard_pressure_hpa(&self, h_km: f64) -> f64 {
        let h_p = 6356.766 * h_km / (6356.766 + h_km);
        if h_p <= 11.0 {
            1013.25 * (288.15 / (288.15 - 6.5 * h_p)).powf(-34.1632 / 6.5)
        } else if h_p <= 20.0 {
            226.3226 * (-34.1632 * (h_p - 11.0) / 216.65).exp()
        } else if h_p <= 32.0 {
            54.74980 * (216.65 / (216.65 + (h_p - 20.0))).powf(34.1632)
        } else if h_p <= 47.0 {
            8.680422 * (228.65 / (228.65 + 2.8 * (h_p - 32.0))).powf(34.1632 / 2.8)
        } else if h_p <= 51.0 {
            1.109106 * (-34.1632 * (h_p - 47.0) / 270.65).exp()
        } else if h_p <= 71.0 {
            0.6694167 * (270.65 / (270.65 - 2.8 * (h_p - 51.0))).powf(-34.1632 / 2.8)
        } else if h_p <= 84.852 {
            0.03956649 * (214.65 / (214.65 - 2.0 * (h_p - 71.0))).powf(-34.1632 / 2.0)
        } else if (86.0..=100.0).contains(&h_km) {
            (95.571899 - 4.011801 * h_km + 6.424731e-2 * h_km.powi(2) - 4.789660e-4 * h_km.powi(3)
                + 1.340543e-6 * h_km.powi(4))
            .exp()
        } else {
            1e-62
        }
    }

    fn standard_temperature_k(&self, h_km: f64) -> f64 {
        let h_p = 6356.766 * h_km / (6356.766 + h_km);
        if h_p <= 11.0 {
            288.15 - 6.5 * h_p
        } else if h_p <= 20.0 {
            216.65
        } else if h_p <= 32.0 {
            216.65 + (h_p - 20.0)
        } else if h_p <= 47.0 {
            228.65 + 2.8 * (h_p - 32.0)
        } else if h_p <= 51.0 {
            270.65
        } else if h_p <= 71.0 {
            270.65 - 2.8 * (h_p - 51.0)
        } else if h_p <= 84.852 {
            214.65 - 2.0 * (h_p - 71.0)
        } else if (86.0..=91.0).contains(&h_km) {
            186.8673
        } else if (91.0..=100.0).contains(&h_km) {
            263.1905 - 76.3232 * (1.0 - ((h_km - 91.0) / 19.9429).powi(2)).sqrt()
        } else {
            195.08134
        }
    }

    fn standard_water_vapour_density_gm3(&self, h_km: f64, rho0_gm3: f64) -> f64 {
        rho0_gm3 * (-h_km / 2.0).exp()
    }

    fn radio_refractive_index(&self, pd_hpa: f64, e_hpa: f64, t_k: f64) -> f64 {
        let n = 77.6 * pd_hpa / t_k + 72.0 * e_hpa / t_k + 3.75e5 * e_hpa / t_k.powi(2);
        1.0 + n * 1e-6
    }

    fn surface_water_vapour_density_gm3(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        p: f64,
        alt_km: f64,
    ) -> f64 {
        self.interpolator_836_scalar(&self.rho_836_v6, lat_deg, lon_deg, p, Some(alt_km))
    }

    fn total_water_vapour_content_kgm2(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        p: f64,
        alt_km: f64,
    ) -> f64 {
        self.interpolator_836_scalar(&self.v_836_v6, lat_deg, lon_deg, p, Some(alt_km))
    }

    fn interpolator_836_scalar(
        &self,
        datasets: &[(f64, RegularGrid2D)],
        lat_deg: f64,
        lon_deg: f64,
        p: f64,
        alt_km: Option<f64>,
    ) -> f64 {
        let lon_mod = mod_360(lon_deg);
        let (p_below, p_above, p_exact) = percentile_bounds(&P836_LEVELS, p);

        let r = ((90.0 - lat_deg) / 1.125).floor();
        let c = (lon_mod / 1.125).floor();

        let lats = [
            90.0 - r * 1.125,
            90.0 - (r + 1.0) * 1.125,
            90.0 - r * 1.125,
            90.0 - (r + 1.0) * 1.125,
        ];
        let lons = [
            mod_360(c * 1.125),
            mod_360(c * 1.125),
            mod_360((c + 1.0) * 1.125),
            mod_360((c + 1.0) * 1.125),
        ];

        let frac_r = (90.0 - lat_deg) / 1.125;
        let frac_c = lon_mod / 1.125;

        let mut altitude_res = [0.0_f64; 4];
        for i in 0..4 {
            altitude_res[i] = self.topo_836_v6.interp(lats[i], lons[i]);
        }

        let alt = alt_km.unwrap_or(0.0);
        let use_alt_scalar = alt_km.is_some();

        let data_a = self.adjust_836_and_blend(
            datasets,
            p_above,
            &lats,
            &lons,
            &altitude_res,
            alt,
            use_alt_scalar,
            frac_r,
            frac_c,
        );
        if p_exact {
            data_a
        } else {
            let data_b = self.adjust_836_and_blend(
                datasets,
                p_below,
                &lats,
                &lons,
                &altitude_res,
                alt,
                use_alt_scalar,
                frac_r,
                frac_c,
            );
            data_b + (data_a - data_b) * (p.ln() - p_below.ln()) / (p_above.ln() - p_below.ln())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn adjust_836_and_blend(
        &self,
        datasets: &[(f64, RegularGrid2D)],
        p: f64,
        lats: &[f64; 4],
        lons: &[f64; 4],
        altitude_res: &[f64; 4],
        alt_scalar: f64,
        use_alt_scalar: bool,
        r: f64,
        c: f64,
    ) -> f64 {
        let data_grid = grid_for_p(datasets, p);
        let vsch_grid = grid_for_p(&self.vsch_836_v6, p);

        let r_floor = r.floor();
        let c_floor = c.floor();
        let weights = [
            (r_floor + 1.0 - r) * (c_floor + 1.0 - c),
            (r - r_floor) * (c_floor + 1.0 - c),
            (r_floor + 1.0 - r) * (c - c_floor),
            (r - r_floor) * (c - c_floor),
        ];

        let mut blended = 0.0;
        for i in 0..4 {
            let base = data_grid.interp(lats[i], lons[i]);
            let vsch = vsch_grid.interp(lats[i], lons[i]);
            let alt_here = if use_alt_scalar {
                alt_scalar
            } else {
                altitude_res[i]
            };
            let adjusted = base * (-(alt_here - altitude_res[i]) / vsch).exp();
            blended += adjusted * weights[i];
        }
        blended
    }

    fn rainfall_rate_r001_mmh(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.rainfall_r001_837_v7
            .interp(lat_deg, wrap_lon_180(lon_deg))
    }

    fn rain_height_km(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.zero_isotherm_839_v4.interp(lat_deg, mod_360(lon_deg)) + 0.36
    }

    fn cloud_reduced_liquid_kgm2(&self, lat_deg: f64, lon_deg: f64, p: f64) -> f64 {
        interpolate_grid_log_p(
            &self.cloud_lred_840_v9,
            &P840_LEVELS,
            lat_deg,
            mod_360(lon_deg),
            p,
        )
    }

    fn map_wet_term_radio_refractivity(&self, lat_deg: f64, lon_deg: f64, p: f64) -> f64 {
        interpolate_grid_log_p(
            &self.wet_refractivity_453_v13,
            &P453_LEVELS,
            lat_deg,
            wrap_lon_180(lon_deg),
            p,
        )
    }

    fn gamma0_exact_v13(&self, freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 {
        let theta = 300.0 / temp_k;
        let e = rho_gm3 * temp_k / 216.7;

        let mut n_pp = 0.0;
        for line in &self.oxygen_lines_v13 {
            let d_f = line.c3 * 1e-4 * (pressure_hpa * theta.powf(0.8 - line.c4) + 1.1 * e * theta);
            let d_f = (d_f * d_f + 2.25e-6).sqrt();
            let delta = (line.c5 + line.c6 * theta) * 1e-4 * (pressure_hpa + e) * theta.powf(0.8);
            let f_i = freq_ghz / line.f
                * ((d_f - delta * (line.f - freq_ghz)) / ((line.f - freq_ghz).powi(2) + d_f * d_f)
                    + (d_f - delta * (line.f + freq_ghz))
                        / ((line.f + freq_ghz).powi(2) + d_f * d_f));
            let s_i =
                line.c1 * 1e-7 * pressure_hpa * theta.powi(3) * (line.c2 * (1.0 - theta)).exp();
            n_pp += s_i * f_i;
        }

        let d = 5.6e-4 * (pressure_hpa + e) * theta.powf(0.8);
        let n_d_pp = freq_ghz
            * pressure_hpa
            * theta.powi(2)
            * (6.14e-5 / (d * (1.0 + (freq_ghz / d).powi(2)))
                + 1.4e-12 * pressure_hpa * theta.powf(1.5) / (1.0 + 1.9e-5 * freq_ghz.powf(1.5)));

        0.1820 * freq_ghz * (n_pp + n_d_pp)
    }

    fn gammaw_exact_v13(&self, freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 {
        let theta = 300.0 / temp_k;
        let e = rho_gm3 * temp_k / 216.7;

        let mut n_pp = 0.0;
        for line in &self.water_lines_v13 {
            let d_f = line.c3
                * 1e-4
                * (pressure_hpa * theta.powf(line.c4) + line.c5 * e * theta.powf(line.c6));
            let d_f =
                0.535 * d_f + (0.217 * d_f * d_f + 2.1316e-12 * line.f * line.f / theta).sqrt();
            let f_i = freq_ghz / line.f
                * (d_f / ((line.f - freq_ghz).powi(2) + d_f * d_f)
                    + d_f / ((line.f + freq_ghz).powi(2) + d_f * d_f));
            let s_i = line.c1 * 1e-1 * e * theta.powf(3.5) * (line.c2 * (1.0 - theta)).exp();
            n_pp += s_i * f_i;
        }

        0.1820 * freq_ghz * n_pp
    }

    fn gamma_exact_v13(&self, freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 {
        self.gamma0_exact_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k)
            + self.gammaw_exact_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k)
    }

    fn slant_inclined_path_equivalent_height_v13(
        &self,
        freq_ghz: f64,
        pressure_hpa: f64,
        rho_gm3: f64,
        temp_k: f64,
    ) -> (f64, f64) {
        let e = rho_gm3 * temp_k / 216.7;
        let ps = pressure_hpa + e;
        let a0 = interpolate_h0_coeff(&self.h0_coeffs_v13, freq_ghz, |c| c.a0);
        let b0 = interpolate_h0_coeff(&self.h0_coeffs_v13, freq_ghz, |c| c.b0);
        let c0 = interpolate_h0_coeff(&self.h0_coeffs_v13, freq_ghz, |c| c.c0);
        let d0 = interpolate_h0_coeff(&self.h0_coeffs_v13, freq_ghz, |c| c.d0);
        let h0 = a0 + b0 * temp_k + c0 * ps + d0 * rho_gm3;

        let hw = HW_A_V13 * freq_ghz
            + HW_B_V13
            + HW_COEFFS_V13
                .iter()
                .map(|(fi, ai, bi)| ai / ((freq_ghz - fi).powi(2) + bi))
                .sum::<f64>();
        (h0, hw)
    }

    fn zenith_water_vapour_attenuation_db(&self, freq_ghz: f64, v_t_kgm2: f64, h_km: f64) -> f64 {
        let f_ref = 20.6;
        let p_ref = 845.0;
        let rho_ref = v_t_kgm2 / 2.38;
        let t_ref_c = 14.0 * (0.22 * v_t_kgm2 / 2.38).ln() + 3.0;

        let a = 0.2048 * (-((freq_ghz - 22.43) / 3.097).powi(2)).exp()
            + 0.2326 * (-((freq_ghz - 183.5) / 4.096).powi(2)).exp()
            + 0.2073 * (-((freq_ghz - 325.0) / 3.651).powi(2)).exp()
            - 0.1113;
        let b = 8.741e4 * (-0.587 * freq_ghz).exp() + 312.2 * freq_ghz.powf(-2.38) + 0.723;
        let h_clipped = h_km.clamp(0.0, 4.0);

        let gamma_ratio = self.gammaw_exact_v13(freq_ghz, p_ref, rho_ref, t_ref_c + 273.15)
            / self.gammaw_exact_v13(f_ref, p_ref, rho_ref, t_ref_c + 273.15);
        let aw_term1 = 0.0176 * v_t_kgm2 * gamma_ratio;
        if freq_ghz < 20.0 {
            aw_term1
        } else {
            aw_term1 * (a * h_clipped.powf(b) + 1.0)
        }
    }

    fn gaseous_attenuation_slant_path_v13(
        &self,
        freq_ghz: f64,
        elevation_deg: f64,
        rho_gm3: f64,
        pressure_hpa: f64,
        temp_k: f64,
        v_t_kgm2: f64,
        h_km: f64,
        exact: bool,
    ) -> f64 {
        if !exact {
            let gamma0 = self.gamma0_exact_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k);
            let gammaw = self.gammaw_exact_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k);
            let (h0, hw) = self.slant_inclined_path_equivalent_height_v13(
                freq_ghz,
                pressure_hpa,
                rho_gm3,
                temp_k,
            );
            let aw = if v_t_kgm2.is_finite() && h_km.is_finite() {
                self.zenith_water_vapour_attenuation_db(freq_ghz, v_t_kgm2, h_km)
            } else {
                gammaw * hw
            };
            let a0 = gamma0 * h0;
            return (a0 + aw) / elevation_deg.to_radians().sin();
        }

        let exp_step = (1.0_f64 / 100.0).exp();
        let denom = exp_step - 1.0;

        let mut n_values = Vec::with_capacity(EXACT_GAS_LAYERS);
        let mut layer_data = Vec::with_capacity(EXACT_GAS_LAYERS);
        for idx in 0..EXACT_GAS_LAYERS {
            let k = idx as f64;
            let delta_h = 0.0001 * (k / 100.0).exp();
            let h_n = 0.0001 * (((k / 100.0).exp() - 1.0) / denom);
            let h_mid = h_n + delta_h / 2.0;
            let t_n = self.standard_temperature_k(h_mid);
            let press_n = self.standard_pressure_hpa(h_mid);
            let rho_n = self.standard_water_vapour_density_gm3(h_mid, rho_gm3);
            let e_n = rho_n * t_n / 216.7;
            let n_n = self.radio_refractive_index(press_n, e_n, t_n);
            n_values.push(n_n);
            layer_data.push((t_n, press_n, rho_n, delta_h, 6371.0 + h_n));
        }

        let mut b = FRAC_PI_2 - elevation_deg.to_radians();
        let mut attenuation_db = 0.0;
        for idx in 0..EXACT_GAS_LAYERS {
            let (t_n, press_n, rho_n, delta_h, r_n) = layer_data[idx];
            let n_ratio = if idx + 1 < EXACT_GAS_LAYERS {
                n_values[idx] / n_values[idx + 1]
            } else {
                1.0
            };

            let cos_b = b.cos();
            let a = -r_n * cos_b
                + 0.5
                    * (4.0 * r_n.powi(2) * cos_b.powi(2)
                        + 8.0 * r_n * delta_h
                        + 4.0 * delta_h.powi(2))
                    .sqrt();
            let alpha = (((r_n / (r_n + delta_h)) * b.sin()).clamp(-1.0, 1.0)).asin();
            let p_dry = press_n - rho_n * t_n / 216.7;
            let gamma = self.gamma_exact_v13(freq_ghz, p_dry, rho_n, t_n);
            attenuation_db += a * gamma;
            b = (alpha.sin() * n_ratio).clamp(-1.0, 1.0).asin();
        }

        attenuation_db
    }

    fn rain_specific_attenuation_coefficients(
        &self,
        freq_ghz: f64,
        elevation_deg: f64,
        tau_deg: f64,
    ) -> (f64, f64) {
        let kh_aj = [-5.33980, -0.35351, -0.23789, -0.94158];
        let kh_bj = [-0.10008, 1.2697, 0.86036, 0.64552];
        let kh_cj = [1.13098, 0.454, 0.15354, 0.16817];
        let kv_aj = [-3.80595, -3.44965, -0.39902, 0.50167];
        let kv_bj = [0.56934, -0.22911, 0.73042, 1.07319];
        let kv_cj = [0.81061, 0.51059, 0.11899, 0.27195];
        let ah_aj = [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721];
        let ah_bj = [1.82442, 0.77564, 0.63773, -0.96230, -3.29980];
        let ah_cj = [-0.55187, 0.19822, 0.13164, 1.47828, 3.4399];
        let av_aj = [-0.07771, 0.56727, -0.20238, -48.2991, 48.5833];
        let av_bj = [2.3384, 0.95545, 1.1452, 0.791669, 0.791459];
        let av_cj = [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479];

        let curve = |f: f64, a: f64, b: f64, c: f64| a * (-((f.log10() - b) / c).powi(2)).exp();

        let kh = 10_f64.powf(
            kh_aj
                .iter()
                .zip(kh_bj.iter())
                .zip(kh_cj.iter())
                .map(|((a, b), c)| curve(freq_ghz, *a, *b, *c))
                .sum::<f64>()
                + (-0.18961) * freq_ghz.log10()
                + 0.71147,
        );
        let kv = 10_f64.powf(
            kv_aj
                .iter()
                .zip(kv_bj.iter())
                .zip(kv_cj.iter())
                .map(|((a, b), c)| curve(freq_ghz, *a, *b, *c))
                .sum::<f64>()
                + (-0.16398) * freq_ghz.log10()
                + 0.63297,
        );

        let alpha_h = ah_aj
            .iter()
            .zip(ah_bj.iter())
            .zip(ah_cj.iter())
            .map(|((a, b), c)| curve(freq_ghz, *a, *b, *c))
            .sum::<f64>()
            + 0.67849 * freq_ghz.log10()
            - 1.95537;
        let alpha_v = av_aj
            .iter()
            .zip(av_bj.iter())
            .zip(av_cj.iter())
            .map(|((a, b), c)| curve(freq_ghz, *a, *b, *c))
            .sum::<f64>()
            + (-0.053739) * freq_ghz.log10()
            + 0.83433;

        let elevation_rad = elevation_deg.to_radians();
        let tau_rad = tau_deg.to_radians();
        let k = (kh + kv + (kh - kv) * elevation_rad.cos().powi(2) * (2.0 * tau_rad).cos()) / 2.0;
        let alpha = (kh * alpha_h
            + kv * alpha_v
            + (kh * alpha_h - kv * alpha_v) * elevation_rad.cos().powi(2) * (2.0 * tau_rad).cos())
            / (2.0 * k);
        (k, alpha)
    }

    fn rain_specific_attenuation_db_per_km(
        &self,
        rainfall_rate_mmh: f64,
        freq_ghz: f64,
        elevation_deg: f64,
        tau_deg: f64,
    ) -> f64 {
        let (k, alpha) =
            self.rain_specific_attenuation_coefficients(freq_ghz, elevation_deg, tau_deg);
        k * rainfall_rate_mmh.powf(alpha)
    }

    fn rain_attenuation_db(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        freq_ghz: f64,
        elevation_deg: f64,
        hs_km: f64,
        p: f64,
        r001_mmh: Option<f64>,
        tau_deg: f64,
        l_s_km: Option<f64>,
    ) -> f64 {
        let re_km = 8500.0;
        let hr_km = self.rain_height_km(lat_deg, lon_deg);

        let elevation_rad = elevation_deg.to_radians();
        let l_s = if let Some(path_km) = l_s_km {
            path_km
        } else if elevation_deg >= 5.0 {
            (hr_km - hs_km) / elevation_rad.sin()
        } else {
            2.0 * (hr_km - hs_km)
                / ((elevation_rad.sin().powi(2) + 2.0 * (hr_km - hs_km) / re_km).sqrt()
                    + elevation_rad.sin())
        };

        let l_g = (l_s * elevation_rad.cos()).abs();
        let r001 =
            r001_mmh.unwrap_or_else(|| self.rainfall_rate_r001_mmh(lat_deg, lon_deg) + EPSILON);
        let gamma_r =
            self.rain_specific_attenuation_db_per_km(r001, freq_ghz, elevation_deg, tau_deg);
        let r001_factor = 1.0
            / (1.0 + 0.78 * (l_g * gamma_r / freq_ghz).sqrt() - 0.38 * (1.0 - (-2.0 * l_g).exp()));

        let eta = (hr_km - hs_km).atan2(l_g * r001_factor).to_degrees();
        let delta_h = if hr_km - hs_km <= 0.0 {
            EPSILON
        } else {
            hr_km - hs_km
        };
        let l_r = if eta > elevation_deg {
            l_g * r001_factor / elevation_rad.cos()
        } else {
            delta_h / elevation_rad.sin()
        };

        let xi = if lat_deg.abs() < 36.0 {
            36.0 - lat_deg.abs()
        } else {
            0.0
        };
        let v001 = 1.0
            / (1.0
                + elevation_rad.sin().sqrt()
                    * (31.0
                        * (1.0 - (-(elevation_deg / (1.0 + xi))).exp())
                        * (l_r * gamma_r).sqrt()
                        / freq_ghz.powi(2)
                        - 0.45));

        let l_e = l_r * v001;
        let a001 = gamma_r * l_e;

        let beta = if p >= 1.0 || lat_deg.abs() >= 36.0 {
            0.0
        } else if elevation_deg > 25.0 {
            -0.005 * (lat_deg.abs() - 36.0)
        } else {
            -0.005 * (lat_deg.abs() - 36.0) + 1.8 - 4.25 * elevation_rad.sin()
        };

        a001 * (p / 0.01).powf(
            -(0.655 + 0.033 * p.ln() - 0.045 * a001.ln() - beta * (1.0 - p) * elevation_rad.sin()),
        )
    }

    fn cloud_liquid_mass_absorption_coefficient(&self, freq_ghz: f64) -> f64 {
        let t_ref_c = 273.75 - 273.15;
        let kl = self.cloud_specific_attenuation_coefficients(freq_ghz, t_ref_c);
        let correction = 0.1522 * (-(freq_ghz + 23.9589).powi(2) / 3.2991e3).exp()
            + 11.51 * (-(freq_ghz - 219.2096).powi(2) / 2.7595e6).exp()
            - 10.4912;
        kl * correction
    }

    fn cloud_specific_attenuation_coefficients(&self, freq_ghz: f64, t_c: f64) -> f64 {
        let t_kelvin = t_c + 273.15;
        let theta = 300.0 / t_kelvin;
        let epsilon0 = 77.66 + 103.3 * (theta - 1.0);
        let epsilon1 = 0.0671 * epsilon0;
        let epsilon2 = 3.52;
        let fp = 20.20 - 146.0 * (theta - 1.0) + 316.0 * (theta - 1.0).powi(2);
        let fs = 39.8 * fp;
        let epsilonp = (epsilon0 - epsilon1) / (1.0 + (freq_ghz / fp).powi(2))
            + (epsilon1 - epsilon2) / (1.0 + (freq_ghz / fs).powi(2))
            + epsilon2;
        let epsilonpp = freq_ghz * (epsilon0 - epsilon1) / (fp * (1.0 + (freq_ghz / fp).powi(2)))
            + freq_ghz * (epsilon1 - epsilon2) / (fs * (1.0 + (freq_ghz / fs).powi(2)));
        let eta = (2.0 + epsilonp) / epsilonpp;
        (0.819 * freq_ghz) / (epsilonpp * (1.0 + eta.powi(2)))
    }

    fn cloud_attenuation_db(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        elevation_deg: f64,
        freq_ghz: f64,
        p: f64,
        lred_kgm2: Option<f64>,
    ) -> f64 {
        let kl = self.cloud_liquid_mass_absorption_coefficient(freq_ghz);
        let lred = lred_kgm2.unwrap_or_else(|| self.cloud_reduced_liquid_kgm2(lat_deg, lon_deg, p));
        (lred * kl / elevation_deg.to_radians().sin()).max(0.0)
    }

    fn wet_term_radio_refractivity(&self, e_hpa: f64, t_c: f64) -> f64 {
        let t_k = t_c + 273.15;
        72.0 * e_hpa / t_k + 3.75e5 * e_hpa / t_k.powi(2)
    }

    fn water_vapour_pressure_hpa(&self, t_c: f64, pressure_hpa: f64, humidity_percent: f64) -> f64 {
        let ef = 1.0 + 1e-4 * (7.2 + pressure_hpa * (0.0320 + 5.9e-6 * t_c.powi(2)));
        let e_s = ef * 6.1121 * (((18.678 - t_c / 234.5) * t_c) / (t_c + 257.14)).exp();
        humidity_percent * e_s / 100.0
    }

    fn scintillation_sigma_db(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        freq_ghz: f64,
        elevation_deg: f64,
        dish_m: f64,
        eta: f64,
        temp_c: Option<f64>,
        humidity_percent: Option<f64>,
        pressure_hpa: Option<f64>,
        h_l_m: f64,
    ) -> f64 {
        let n_wet =
            if let (Some(t_c), Some(h), Some(p_hpa)) = (temp_c, humidity_percent, pressure_hpa) {
                let e = self.water_vapour_pressure_hpa(t_c, p_hpa, h);
                self.wet_term_radio_refractivity(e, t_c)
            } else {
                self.map_wet_term_radio_refractivity(lat_deg, lon_deg, 50.0)
            };

        let sigma_ref = 3.6e-3 + 1e-4 * n_wet;
        let elevation_rad = elevation_deg.to_radians();
        let l =
            2.0 * h_l_m / ((elevation_rad.sin().powi(2) + 2.35e-4).sqrt() + elevation_rad.sin());
        let d_eff = eta.sqrt() * dish_m;
        let x = 1.22 * d_eff.powi(2) * freq_ghz / l;
        let g = if x >= 7.0 {
            0.0
        } else {
            (3.86 * (x.powi(2) + 1.0).powf(11.0 / 12.0) * ((11.0 / 6.0) * 1.0_f64.atan2(x)).sin()
                - 7.08 * x.powf(5.0 / 6.0))
            .sqrt()
        };

        sigma_ref * freq_ghz.powf(7.0 / 12.0) * g / elevation_rad.sin().powf(1.2)
    }

    fn scintillation_attenuation_db(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        freq_ghz: f64,
        elevation_deg: f64,
        p: f64,
        dish_m: f64,
        eta: f64,
        temp_c: Option<f64>,
        humidity_percent: Option<f64>,
        pressure_hpa: Option<f64>,
        h_l_m: f64,
    ) -> f64 {
        let sigma = self.scintillation_sigma_db(
            lat_deg,
            lon_deg,
            freq_ghz,
            elevation_deg,
            dish_m,
            eta,
            temp_c,
            humidity_percent,
            pressure_hpa,
            h_l_m,
        );
        let log_p = p.log10();
        let a = -0.061 * log_p.powi(3) + 0.072 * log_p.powi(2) - 1.71 * log_p + 3.0;
        a * sigma
    }

    fn atmospheric_attenuation(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        freq_ghz: f64,
        elevation_deg: f64,
        p: f64,
        dish_m: f64,
        options: SlantPathOptions,
    ) -> SlantPathContributions {
        let hs_km = options
            .hs_km
            .unwrap_or_else(|| self.topographic_altitude_km(lat_deg, lon_deg));
        let surface_temp_k = self.surface_mean_temperature_k(lat_deg, lon_deg);
        let gas_temp_k = options.t.unwrap_or(surface_temp_k);
        let pressure_hpa = options
            .pressure_hpa
            .unwrap_or_else(|| self.standard_pressure_hpa(hs_km));
        let p_c_g = p.max(1.0);
        let v_t_kgm2 = options.v_t_kgm2.unwrap_or_else(|| {
            self.total_water_vapour_content_kgm2(lat_deg, lon_deg, p_c_g, hs_km)
        });
        let rho_gm3 = options.rho_gm3.unwrap_or_else(|| {
            self.surface_water_vapour_density_gm3(lat_deg, lon_deg, p_c_g, hs_km)
        });

        let rain_db = if options.include_rain {
            self.rain_attenuation_db(
                lat_deg,
                lon_deg,
                freq_ghz,
                elevation_deg,
                hs_km,
                p,
                options.r001_mmh,
                options.tau_deg,
                options.l_s_km,
            )
        } else {
            0.0
        };

        let gas_db = if options.include_gas {
            self.gaseous_attenuation_slant_path_v13(
                freq_ghz,
                elevation_deg,
                rho_gm3,
                pressure_hpa,
                gas_temp_k,
                v_t_kgm2,
                hs_km,
                options.exact,
            )
        } else {
            0.0
        };

        let cloud_db = if options.include_clouds {
            self.cloud_attenuation_db(lat_deg, lon_deg, elevation_deg, freq_ghz, p_c_g, None)
        } else {
            0.0
        };

        let scintillation_temp_c = if options.h_percent.is_some() {
            Some(options.t.unwrap_or(surface_temp_k - 273.15))
        } else {
            None
        };
        let scintillation_pressure_hpa = if options.h_percent.is_some() {
            Some(pressure_hpa)
        } else {
            None
        };
        let scintillation_db = if options.include_scintillation {
            self.scintillation_attenuation_db(
                lat_deg,
                lon_deg,
                freq_ghz,
                elevation_deg,
                p,
                dish_m,
                options.eta,
                scintillation_temp_c,
                options.h_percent,
                scintillation_pressure_hpa,
                options.h_l_m,
            )
        } else {
            0.0
        };

        let total_db = gas_db + ((rain_db + cloud_db).powi(2) + scintillation_db.powi(2)).sqrt();
        SlantPathContributions {
            gas_db,
            cloud_db,
            rain_db,
            scintillation_db,
            total_db,
        }
    }

    fn atmospheric_attenuation_default_gas_only(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        freq_ghz: f64,
        elevation_deg: f64,
        p: f64,
        dish_m: f64,
    ) -> f64 {
        self.atmospheric_attenuation(
            lat_deg,
            lon_deg,
            freq_ghz,
            elevation_deg,
            p,
            dish_m,
            SlantPathOptions {
                hs_km: None,
                rho_gm3: None,
                r001_mmh: None,
                eta: 0.5,
                t: None,
                h_percent: None,
                pressure_hpa: None,
                h_l_m: 1000.0,
                l_s_km: None,
                tau_deg: 45.0,
                v_t_kgm2: None,
                exact: false,
                include_rain: false,
                include_gas: true,
                include_scintillation: false,
                include_clouds: false,
            },
        )
        .total_db
    }
}

fn model() -> Result<&'static IturModel, String> {
    MODEL
        .get_or_init(IturModel::load)
        .as_ref()
        .map_err(|err| err.clone())
}

fn data_root() -> PathBuf {
    std::env::var_os("ITU_RS_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("data"))
}

struct DataBlob {
    label: String,
    bytes: Cow<'static, [u8]>,
}

fn load_data(rel_path: &str) -> Result<DataBlob, String> {
    if let Some(root) = std::env::var_os("ITU_RS_DATA_DIR") {
        let full_path = PathBuf::from(root).join(rel_path);
        let bytes = std::fs::read(&full_path)
            .map_err(|err| format!("failed opening {}: {err}", full_path.display()))?;
        return Ok(DataBlob {
            label: full_path.display().to_string(),
            bytes: Cow::Owned(bytes),
        });
    }

    let full_path = data_root().join(rel_path);
    match std::fs::read(&full_path) {
        Ok(bytes) => {
            return Ok(DataBlob {
                label: full_path.display().to_string(),
                bytes: Cow::Owned(bytes),
            });
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(format!("failed opening {}: {err}", full_path.display())),
    }

    if let Some(bytes) = bundled_data::get(rel_path) {
        return Ok(DataBlob {
            label: format!("bundled data/{rel_path}"),
            bytes: Cow::Borrowed(bytes),
        });
    }

    Err(format!(
        "failed locating data/{rel_path}; set ITU_RS_DATA_DIR to a python-itu-r itur/data directory or enable the itu-rs `data` feature"
    ))
}

fn load_npz_array2(rel_path: &str) -> Result<Array2<f64>, String> {
    let data = load_data(rel_path)?;
    let mut npz = NpzReader::new(Cursor::new(data.bytes.as_ref()))
        .map_err(|err| format!("failed reading npz {}: {err}", data.label))?;
    npz.by_name("arr_0.npy")
        .map_err(|err| format!("failed loading arr_0.npy from {}: {err}", data.label))
}

fn load_spectral_lines(rel_path: &str) -> Result<Vec<SpectralLine>, String> {
    let data = load_data(rel_path)?;
    let reader = BufReader::new(Cursor::new(data.bytes.as_ref()));
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|err| format!("failed reading {} line {}: {err}", data.label, idx + 1))?;
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|part| part.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("failed parsing {} line {}: {err}", data.label, idx + 1))?;
        if cols.len() != 7 {
            return Err(format!(
                "unexpected column count in {} line {}",
                data.label,
                idx + 1
            ));
        }
        out.push(SpectralLine {
            f: cols[0],
            c1: cols[1],
            c2: cols[2],
            c3: cols[3],
            c4: cols[4],
            c5: cols[5],
            c6: cols[6],
        });
    }
    Ok(out)
}

fn load_h0_coefficients(rel_path: &str) -> Result<Vec<H0Coefficients>, String> {
    let data = load_data(rel_path)?;
    let reader = BufReader::new(Cursor::new(data.bytes.as_ref()));
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|err| format!("failed reading {} line {}: {err}", data.label, idx + 1))?;
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|part| part.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("failed parsing {} line {}: {err}", data.label, idx + 1))?;
        if cols.len() != 5 {
            return Err(format!(
                "unexpected column count in {} line {}",
                data.label,
                idx + 1
            ));
        }
        out.push(H0Coefficients {
            freq_ghz: cols[0],
            a0: cols[1],
            b0: cols[2],
            c0: cols[3],
            d0: cols[4],
        });
    }
    Ok(out)
}

fn kernel(d: f64) -> f64 {
    let d = d.abs();
    if d <= 1.0 {
        1.5 * d.powi(3) - 2.5 * d.powi(2) + 1.0
    } else if d <= 2.0 {
        -0.5 * d.powi(3) + 2.5 * d.powi(2) - 4.0 * d + 2.0
    } else {
        0.0
    }
}

fn bracket(axis: &[f64], x: f64) -> (usize, usize, f64) {
    debug_assert!(axis.len() >= 2);
    if x <= axis[0] {
        return (0, 1, 0.0);
    }
    if x >= axis[axis.len() - 1] {
        return (axis.len() - 2, axis.len() - 1, 1.0);
    }

    let hi = searchsorted_right(axis, x);
    let lo = hi - 1;
    let frac = (x - axis[lo]) / (axis[hi] - axis[lo]);
    (lo, hi, frac)
}

fn searchsorted_left(axis: &[f64], x: f64) -> usize {
    axis.partition_point(|value| *value < x)
}

fn searchsorted_right(axis: &[f64], x: f64) -> usize {
    axis.partition_point(|value| *value <= x)
}

fn mod_360(lon_deg: f64) -> f64 {
    lon_deg.rem_euclid(360.0)
}

fn wrap_lon_180(lon_deg: f64) -> f64 {
    let lon_mod = mod_360(lon_deg);
    if lon_mod > 180.0 {
        lon_mod - 360.0
    } else {
        lon_mod
    }
}

fn p836_suffix(p: f64) -> String {
    let mut s = p.to_string();
    s.retain(|ch| ch != '.');
    s
}

fn p453_suffix(p: f64) -> String {
    let mut s = p.to_string();
    s.retain(|ch| ch != '.');
    s
}

fn p840_stem(p: f64) -> Result<&'static str, String> {
    match p {
        x if (x - 0.01).abs() < 1e-12 => Ok("001"),
        x if (x - 0.02).abs() < 1e-12 => Ok("002"),
        x if (x - 0.03).abs() < 1e-12 => Ok("003"),
        x if (x - 0.05).abs() < 1e-12 => Ok("005"),
        x if (x - 0.1).abs() < 1e-12 => Ok("01"),
        x if (x - 0.2).abs() < 1e-12 => Ok("02"),
        x if (x - 0.3).abs() < 1e-12 => Ok("03"),
        x if (x - 0.5).abs() < 1e-12 => Ok("05"),
        x if (x - 1.0).abs() < 1e-12 => Ok("1"),
        x if (x - 2.0).abs() < 1e-12 => Ok("2"),
        x if (x - 3.0).abs() < 1e-12 => Ok("3"),
        x if (x - 5.0).abs() < 1e-12 => Ok("5"),
        x if (x - 10.0).abs() < 1e-12 => Ok("10"),
        x if (x - 20.0).abs() < 1e-12 => Ok("20"),
        x if (x - 30.0).abs() < 1e-12 => Ok("30"),
        x if (x - 50.0).abs() < 1e-12 => Ok("50"),
        x if (x - 60.0).abs() < 1e-12 => Ok("60"),
        x if (x - 70.0).abs() < 1e-12 => Ok("70"),
        x if (x - 80.0).abs() < 1e-12 => Ok("80"),
        x if (x - 90.0).abs() < 1e-12 => Ok("90"),
        x if (x - 95.0).abs() < 1e-12 => Ok("95"),
        x if (x - 99.0).abs() < 1e-12 => Ok("99"),
        x if (x - 100.0).abs() < 1e-12 => Ok("100"),
        _ => Err(format!("unsupported P.840 percentile {p}")),
    }
}

fn percentile_bounds(levels: &[f64], p: f64) -> (f64, f64, bool) {
    for level in levels {
        if (p - *level).abs() < 1e-12 {
            return (*level, *level, true);
        }
    }

    let insertion = levels.partition_point(|level| *level < p);
    if insertion == 0 {
        (levels[0], levels[1], false)
    } else if insertion >= levels.len() {
        (levels[levels.len() - 2], levels[levels.len() - 1], false)
    } else {
        (levels[insertion - 1], levels[insertion], false)
    }
}

fn grid_for_p(datasets: &[(f64, RegularGrid2D)], p: f64) -> &RegularGrid2D {
    datasets
        .iter()
        .find(|(level, _)| (*level - p).abs() < 1e-12)
        .map(|(_, grid)| grid)
        .expect("missing percentile dataset")
}

fn interpolate_grid_log_p(
    datasets: &[(f64, RegularGrid2D)],
    levels: &[f64],
    lat_deg: f64,
    lon_deg: f64,
    p: f64,
) -> f64 {
    let (p_below, p_above, p_exact) = percentile_bounds(levels, p);
    let above = grid_for_p(datasets, p_above).interp(lat_deg, lon_deg);
    if p_exact {
        above
    } else {
        let below = grid_for_p(datasets, p_below).interp(lat_deg, lon_deg);
        below + (above - below) * (p.ln() - p_below.ln()) / (p_above.ln() - p_below.ln())
    }
}

fn interpolate_h0_coeff(
    coeffs: &[H0Coefficients],
    freq_ghz: f64,
    map: impl Fn(&H0Coefficients) -> f64,
) -> f64 {
    if freq_ghz <= coeffs[0].freq_ghz {
        return map(&coeffs[0]);
    }
    if freq_ghz >= coeffs[coeffs.len() - 1].freq_ghz {
        return map(&coeffs[coeffs.len() - 1]);
    }

    let hi = coeffs.partition_point(|entry| entry.freq_ghz < freq_ghz);
    let lo = hi - 1;
    let x0 = coeffs[lo].freq_ghz;
    let x1 = coeffs[hi].freq_ghz;
    let y0 = map(&coeffs[lo]);
    let y1 = map(&coeffs[hi]);
    y0 + (y1 - y0) * (freq_ghz - x0) / (x1 - x0)
}

fn validate_common_inputs(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    p: f64,
    dish_m: f64,
) -> Result<(), String> {
    if !lat_deg.is_finite()
        || !lon_deg.is_finite()
        || !freq_ghz.is_finite()
        || !p.is_finite()
        || !dish_m.is_finite()
    {
        return Err("all required inputs must be finite".to_string());
    }
    if !(-90.0..=90.0).contains(&lat_deg) {
        return Err("lat_deg must be in [-90, 90]".to_string());
    }
    if freq_ghz <= 0.0 {
        return Err("freq_ghz must be > 0".to_string());
    }
    if p <= 0.0 {
        return Err("p must be > 0".to_string());
    }
    if dish_m <= 0.0 {
        return Err("d_m must be > 0".to_string());
    }
    Ok(())
}

fn validate_elevation_deg(elevation_deg: f64) -> Result<(), String> {
    if !elevation_deg.is_finite() {
        return Err("elevation_deg must be finite".to_string());
    }
    if elevation_deg <= 0.0 || elevation_deg >= 90.0 {
        return Err("elevation_deg must be in (0, 90)".to_string());
    }
    Ok(())
}

fn validate_lat_lon(lat_deg: f64, lon_deg: f64) -> Result<(), String> {
    if !lat_deg.is_finite() || !lon_deg.is_finite() {
        return Err("lat_deg and lon_deg must be finite".to_string());
    }
    if !(-90.0..=90.0).contains(&lat_deg) {
        return Err("lat_deg must be in [-90, 90]".to_string());
    }
    Ok(())
}

fn validate_positive(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    if value <= 0.0 {
        return Err(format!("{name} must be > 0"));
    }
    Ok(())
}

fn validate_nonnegative(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    if value < 0.0 {
        return Err(format!("{name} must be >= 0"));
    }
    Ok(())
}

fn validate_finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn validate_p(p: f64) -> Result<(), String> {
    validate_positive("p", p)
}

fn validate_tau_deg(tau_deg: f64) -> Result<(), String> {
    validate_finite("tau_deg", tau_deg)?;
    if !(0.0..=90.0).contains(&tau_deg) {
        return Err("tau_deg must be in [0, 90]".to_string());
    }
    Ok(())
}

fn validate_optional_nonnegative(name: &str, value: Option<f64>) -> Result<(), String> {
    if let Some(value) = value {
        validate_nonnegative(name, value)?;
    }
    Ok(())
}

fn validate_optional_positive(name: &str, value: Option<f64>) -> Result<(), String> {
    if let Some(value) = value {
        validate_positive(name, value)?;
    }
    Ok(())
}

fn validate_options(options: SlantPathOptions) -> Result<(), String> {
    let optional_values = [
        options.hs_km,
        options.rho_gm3,
        options.r001_mmh,
        options.t,
        options.h_percent,
        options.pressure_hpa,
        options.l_s_km,
        options.v_t_kgm2,
    ];
    if optional_values
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("optional numeric inputs must be finite when provided".to_string());
    }
    if !options.eta.is_finite() || !options.h_l_m.is_finite() || !options.tau_deg.is_finite() {
        return Err("eta, h_l_m, and tau_deg must be finite".to_string());
    }
    if options.eta <= 0.0 || options.eta > 1.0 {
        return Err("eta must be in (0, 1]".to_string());
    }
    if options.h_l_m <= 0.0 {
        return Err("h_l_m must be > 0".to_string());
    }
    if !(0.0..=90.0).contains(&options.tau_deg) {
        return Err("tau_deg must be in [0, 90]".to_string());
    }
    if let Some(rho_gm3) = options.rho_gm3
        && rho_gm3 < 0.0
    {
        return Err("rho_gm3 must be >= 0".to_string());
    }
    if let Some(r001_mmh) = options.r001_mmh
        && r001_mmh < 0.0
    {
        return Err("r001_mmh must be >= 0".to_string());
    }
    if let Some(t) = options.t
        && t <= 0.0
    {
        return Err("t must be > 0".to_string());
    }
    if let Some(h_percent) = options.h_percent
        && !(0.0..=100.0).contains(&h_percent)
    {
        return Err("h_percent must be in [0, 100]".to_string());
    }
    if let Some(pressure_hpa) = options.pressure_hpa
        && pressure_hpa <= 0.0
    {
        return Err("pressure_hpa must be > 0".to_string());
    }
    if let Some(l_s_km) = options.l_s_km
        && l_s_km <= 0.0
    {
        return Err("l_s_km must be > 0".to_string());
    }
    if let Some(v_t_kgm2) = options.v_t_kgm2
        && v_t_kgm2 < 0.0
    {
        return Err("v_t_kgm2 must be >= 0".to_string());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rust_itur_slant_path_scalar(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    p: f64,
    dish_m: f64,
    options: SlantPathOptions,
) -> Result<SlantPathContributions, String> {
    validate_common_inputs(lat_deg, lon_deg, freq_ghz, p, dish_m)?;
    validate_elevation_deg(elevation_deg)?;
    validate_options(options)?;
    Ok(model()?.atmospheric_attenuation(
        lat_deg,
        lon_deg,
        freq_ghz,
        elevation_deg,
        p,
        dish_m,
        options,
    ))
}

#[allow(dead_code)]
fn gas_attenuation_default_many_clamped(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: &[f64],
    p: f64,
    dish_m: f64,
) -> Result<Vec<f64>, String> {
    validate_common_inputs(lat_deg, lon_deg, freq_ghz, p, dish_m)?;
    let model = model()?;
    let mut out = Vec::with_capacity(elevation_deg.len());
    for &el in elevation_deg {
        let el_query = el.clamp(0.01, 89.99);
        validate_elevation_deg(el_query)?;
        out.push(model.atmospheric_attenuation_default_gas_only(
            lat_deg, lon_deg, freq_ghz, el_query, p, dish_m,
        ));
    }
    Ok(out)
}

/// Looks up topographic altitude above mean sea level from ITU-R [P.1511](https://www.itu.int/rec/R-REC-P.1511).
///
/// This is the terrain height used when a slant-path calculation needs the
/// Earth station altitude and `SlantPathOptions::hs_km` is not supplied.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
///
/// # Returns
///
/// Terrain height in kilometres above mean sea level. The value is clamped to a
/// tiny positive floor internally so downstream propagation equations do not
/// see an exact zero height.
///
/// # Errors
///
/// Returns an error if coordinates are not finite, latitude is outside
/// `[-90, 90]`, or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let h_km = itu_rs::topographic_altitude_km(45.4215, -75.6972)?;
///
/// assert!(h_km.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn topographic_altitude_km(lat_deg: f64, lon_deg: f64) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg).map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .topographic_altitude_km(lat_deg, lon_deg))
}

/// Looks up annual mean surface temperature from ITU-R [P.1510](https://www.itu.int/rec/R-REC-P.1510).
///
/// This provides the site temperature used by default slant-path gas
/// calculations when `SlantPathOptions::t` is not supplied.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
///
/// # Returns
///
/// Annual mean surface temperature in kelvin.
///
/// # Errors
///
/// Returns an error if coordinates are invalid or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1510/v1_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let temp_k = itu_rs::surface_mean_temperature_k(45.4215, -75.6972)?;
///
/// assert!(temp_k > 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn surface_mean_temperature_k(
    lat_deg: f64,
    lon_deg: f64,
) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg).map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .surface_mean_temperature_k(lat_deg, lon_deg))
}

/// Computes standard-atmosphere temperature from ITU-R [P.835](https://www.itu.int/rec/R-REC-P.835).
///
/// The [P.835](https://www.itu.int/rec/R-REC-P.835) reference atmosphere supplies a temperature profile by geometric
/// height. This is useful when a site-specific temperature is not available or
/// when evaluating gas attenuation at a layer height.
///
/// # Parameters
///
/// - `h_km`: geometric height above mean sea level in kilometres. Must be
///   non-negative.
///
/// # Returns
///
/// Standard-atmosphere temperature in kelvin.
///
/// # Errors
///
/// Returns an error if `h_km` is not finite, is negative, or model data cannot
/// be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let temp_k = itu_rs::standard_temperature_k(1.0)?;
///
/// assert!(temp_k > 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn standard_temperature_k(h_km: f64) -> std::result::Result<f64, ItuError> {
    validate_nonnegative("h_km", h_km).map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .standard_temperature_k(h_km))
}

/// Computes standard-atmosphere pressure from ITU-R [P.835](https://www.itu.int/rec/R-REC-P.835).
///
/// The pressure profile is used by the gas model when surface pressure is not
/// supplied explicitly.
///
/// # Parameters
///
/// - `h_km`: geometric height above mean sea level in kilometres. Must be
///   non-negative.
///
/// # Returns
///
/// Dry-air atmospheric pressure in hPa.
///
/// # Errors
///
/// Returns an error if `h_km` is not finite, is negative, or model data cannot
/// be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let pressure_hpa = itu_rs::standard_pressure_hpa(1.0)?;
///
/// assert!(pressure_hpa > 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn standard_pressure_hpa(h_km: f64) -> std::result::Result<f64, ItuError> {
    validate_nonnegative("h_km", h_km).map_err(ItuError::from)?;
    Ok(model().map_err(ItuError::from)?.standard_pressure_hpa(h_km))
}

/// Computes standard water-vapour density from ITU-R [P.835](https://www.itu.int/rec/R-REC-P.835).
///
/// This applies the [P.835](https://www.itu.int/rec/R-REC-P.835) exponential decrease of water-vapour density with
/// height from a surface reference density.
///
/// # Parameters
///
/// - `h_km`: geometric height above mean sea level in kilometres. Must be
///   non-negative.
/// - `rho0_gm3`: surface water-vapour density in g/m^3. Must be non-negative.
///
/// # Returns
///
/// Water-vapour density at `h_km` in g/m^3.
///
/// # Errors
///
/// Returns an error if either input is not finite, if either input is negative,
/// or if model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let rho_gm3 = itu_rs::standard_water_vapour_density_gm3(1.0, 7.5)?;
///
/// assert!(rho_gm3 >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn standard_water_vapour_density_gm3(
    h_km: f64,
    rho0_gm3: f64,
) -> std::result::Result<f64, ItuError> {
    validate_nonnegative("h_km", h_km)
        .and_then(|_| validate_nonnegative("rho0_gm3", rho0_gm3))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .standard_water_vapour_density_gm3(h_km, rho0_gm3))
}

/// Looks up surface water-vapour density from ITU-R [P.836](https://www.itu.int/rec/R-REC-P.836).
///
/// Surface water-vapour density is near-ground absolute humidity. The slant-path
/// gas model uses this value when `SlantPathOptions::rho_gm3` is not supplied.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `p`: percentage of time exceeded, such as `1.0` for 1%. Must be positive.
/// - `alt_km`: station altitude in kilometres. Finite values are accepted so
///   sites below sea level can be represented.
///
/// # Returns
///
/// Surface water-vapour density in g/m^3.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/836/v6_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let rho_gm3 = itu_rs::surface_water_vapour_density_gm3(
///     45.4215, -75.6972, 1.0, 0.1,
/// )?;
///
/// assert!(rho_gm3.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn surface_water_vapour_density_gm3(
    lat_deg: f64,
    lon_deg: f64,
    p: f64,
    alt_km: f64,
) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg)
        .and_then(|_| validate_p(p))
        .and_then(|_| validate_finite("alt_km", alt_km))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .surface_water_vapour_density_gm3(lat_deg, lon_deg, p, alt_km))
}

/// Looks up total columnar water-vapour content from ITU-R [P.836](https://www.itu.int/rec/R-REC-P.836).
///
/// Total water vapour content is the vertically integrated water-vapour mass
/// above the site. The approximate [P.676](https://www.itu.int/rec/R-REC-P.676) gas path uses this when
/// `SlantPathOptions::v_t_kgm2` is not supplied.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `p`: percentage of time exceeded. Must be positive.
/// - `alt_km`: station altitude in kilometres.
///
/// # Returns
///
/// Total columnar water-vapour content in kg/m^2.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/836/v6_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let vapour_kgm2 = itu_rs::total_water_vapour_content_kgm2(
///     45.4215, -75.6972, 1.0, 0.1,
/// )?;
///
/// assert!(vapour_kgm2.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn total_water_vapour_content_kgm2(
    lat_deg: f64,
    lon_deg: f64,
    p: f64,
    alt_km: f64,
) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg)
        .and_then(|_| validate_p(p))
        .and_then(|_| validate_finite("alt_km", alt_km))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .total_water_vapour_content_kgm2(lat_deg, lon_deg, p, alt_km))
}

/// Looks up rainfall rate exceeded for 0.01% of an average year from ITU-R [P.837](https://www.itu.int/rec/R-REC-P.837).
///
/// This reference rain rate, usually written `R_0.01`, anchors the [P.618](https://www.itu.int/rec/R-REC-P.618) rain
/// fade calculation. Other time percentages are derived from it.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
///
/// # Returns
///
/// Rainfall rate in mm/h exceeded for 0.01% of an average year.
///
/// # Errors
///
/// Returns an error if coordinates are invalid or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/837/v7_lat_r001.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let r001_mmh = itu_rs::rainfall_rate_r001_mmh(45.4215, -75.6972)?;
///
/// assert!(r001_mmh >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn rainfall_rate_r001_mmh(lat_deg: f64, lon_deg: f64) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg).map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .rainfall_rate_r001_mmh(lat_deg, lon_deg))
}

/// Looks up rain height from ITU-R [P.839](https://www.itu.int/rec/R-REC-P.839).
///
/// Rain height approximates the altitude above which rain is no longer present.
/// It is used with station height and elevation angle to determine the slant
/// path length through rain.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
///
/// # Returns
///
/// Rain height in kilometres above mean sea level.
///
/// # Errors
///
/// Returns an error if coordinates are invalid or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/839/v4_esalat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let h_rain_km = itu_rs::rain_height_km(45.4215, -75.6972)?;
///
/// assert!(h_rain_km.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn rain_height_km(lat_deg: f64, lon_deg: f64) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg).map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .rain_height_km(lat_deg, lon_deg))
}

/// Computes ITU-R [P.838](https://www.itu.int/rec/R-REC-P.838) rain specific attenuation coefficients.
///
/// The returned `(k, alpha)` coefficients convert rainfall rate `R` to specific
/// rain attenuation with `gamma_R = k * R^alpha`. Frequency, elevation, and
/// polarization affect drop-shape and polarization coupling in the model.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: path elevation angle above the horizon in degrees, in
///   `(0, 90)`.
/// - `tau_deg`: polarization tilt angle in degrees, in `[0, 90]`.
///
/// # Returns
///
/// `(k, alpha)` [P.838](https://www.itu.int/rec/R-REC-P.838) coefficients for use with rainfall rate in mm/h.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let (k, alpha) = itu_rs::rain_specific_attenuation_coefficients(
///     12.0, 30.0, 45.0,
/// )?;
///
/// assert!(k.is_finite());
/// assert!(alpha.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn rain_specific_attenuation_coefficients(
    freq_ghz: f64,
    elevation_deg: f64,
    tau_deg: f64,
) -> std::result::Result<(f64, f64), ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_elevation_deg(elevation_deg))
        .and_then(|_| validate_tau_deg(tau_deg))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .rain_specific_attenuation_coefficients(freq_ghz, elevation_deg, tau_deg))
}

/// Computes ITU-R [P.838](https://www.itu.int/rec/R-REC-P.838) rain specific attenuation in dB/km.
///
/// This applies the [P.838](https://www.itu.int/rec/R-REC-P.838) relation `gamma_R = k * R^alpha` for a supplied
/// rainfall rate and path geometry.
///
/// # Parameters
///
/// - `rainfall_rate_mmh`: rainfall rate in mm/h. Must be non-negative.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: path elevation angle above the horizon in degrees, in
///   `(0, 90)`.
/// - `tau_deg`: polarization tilt angle in degrees, in `[0, 90]`.
///
/// # Returns
///
/// Specific rain attenuation in dB/km.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let gamma_db_per_km = itu_rs::rain_specific_attenuation_db_per_km(
///     28.0, 12.0, 30.0, 45.0,
/// )?;
///
/// assert!(gamma_db_per_km >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn rain_specific_attenuation_db_per_km(
    rainfall_rate_mmh: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    tau_deg: f64,
) -> std::result::Result<f64, ItuError> {
    validate_nonnegative("rainfall_rate_mmh", rainfall_rate_mmh)
        .and_then(|_| validate_positive("freq_ghz", freq_ghz))
        .and_then(|_| validate_elevation_deg(elevation_deg))
        .and_then(|_| validate_tau_deg(tau_deg))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .rain_specific_attenuation_db_per_km(rainfall_rate_mmh, freq_ghz, elevation_deg, tau_deg))
}

/// Looks up reduced cloud liquid water content from ITU-R [P.840](https://www.itu.int/rec/R-REC-P.840).
///
/// Reduced cloud liquid water content is the cloud liquid column used by the
/// [P.840](https://www.itu.int/rec/R-REC-P.840) cloud attenuation model for the requested time percentage.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `p`: percentage of time exceeded. Must be positive.
///
/// # Returns
///
/// Reduced cloud liquid water content in kg/m^2.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/840/v9_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let lred_kgm2 = itu_rs::cloud_reduced_liquid_kgm2(45.4215, -75.6972, 1.0)?;
///
/// assert!(lred_kgm2 >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn cloud_reduced_liquid_kgm2(
    lat_deg: f64,
    lon_deg: f64,
    p: f64,
) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg)
        .and_then(|_| validate_p(p))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .cloud_reduced_liquid_kgm2(lat_deg, lon_deg, p))
}

/// Computes the [P.840](https://www.itu.int/rec/R-REC-P.840) cloud liquid-water mass absorption coefficient.
///
/// The coefficient describes how efficiently cloud liquid water absorbs radio
/// waves at the supplied frequency before local temperature is applied by the
/// full specific attenuation coefficient.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
///
/// # Returns
///
/// Liquid-water mass absorption coefficient used by [P.840](https://www.itu.int/rec/R-REC-P.840).
///
/// # Errors
///
/// Returns an error if `freq_ghz` is not finite, is not positive, or model data
/// cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let coeff = itu_rs::cloud_liquid_mass_absorption_coefficient(12.0)?;
///
/// assert!(coeff.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn cloud_liquid_mass_absorption_coefficient(
    freq_ghz: f64,
) -> std::result::Result<f64, ItuError> {
    validate_positive("freq_ghz", freq_ghz).map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .cloud_liquid_mass_absorption_coefficient(freq_ghz))
}

/// Computes the [P.840](https://www.itu.int/rec/R-REC-P.840) cloud specific attenuation coefficient.
///
/// This coefficient converts reduced liquid water content into cloud attenuation
/// for a given radio frequency and cloud temperature.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `temp_c`: cloud temperature in degrees Celsius. Must be above absolute
///   zero.
///
/// # Returns
///
/// Cloud specific attenuation coefficient in dB/km per g/m^3 as defined by
/// [P.840](https://www.itu.int/rec/R-REC-P.840) for the implemented model.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let coeff = itu_rs::cloud_specific_attenuation_coefficient(12.0, 0.0)?;
///
/// assert!(coeff.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn cloud_specific_attenuation_coefficient(
    freq_ghz: f64,
    temp_c: f64,
) -> std::result::Result<f64, ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_finite("temp_c", temp_c))
        .and_then(|_| {
            if temp_c <= -273.15 {
                Err("temp_c must be > -273.15".to_string())
            } else {
                Ok(())
            }
        })
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .cloud_specific_attenuation_coefficients(freq_ghz, temp_c))
}

/// Computes cloud attenuation from ITU-R [P.840](https://www.itu.int/rec/R-REC-P.840).
///
/// Cloud attenuation is the path attenuation caused by liquid water in clouds
/// along an Earth-space slant path. If `lred_kgm2` is not supplied, the [P.840](https://www.itu.int/rec/R-REC-P.840)
/// map is used for the requested site and time percentage.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `elevation_deg`: path elevation angle in degrees, in `(0, 90)`.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `p`: percentage of time exceeded. Must be positive.
/// - `lred_kgm2`: optional reduced cloud liquid water content in kg/m^2. When
///   `None`, it is looked up from [P.840](https://www.itu.int/rec/R-REC-P.840).
///
/// # Returns
///
/// Cloud attenuation in dB.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/840/v9_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let cloud_db = itu_rs::cloud_attenuation_db(
///     45.4215, -75.6972, 30.0, 12.0, 1.0, None,
/// )?;
///
/// assert!(cloud_db >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn cloud_attenuation_db(
    lat_deg: f64,
    lon_deg: f64,
    elevation_deg: f64,
    freq_ghz: f64,
    p: f64,
    lred_kgm2: Option<f64>,
) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg)
        .and_then(|_| validate_elevation_deg(elevation_deg))
        .and_then(|_| validate_positive("freq_ghz", freq_ghz))
        .and_then(|_| validate_p(p))
        .and_then(|_| validate_optional_nonnegative("lred_kgm2", lred_kgm2))
        .map_err(ItuError::from)?;
    Ok(model().map_err(ItuError::from)?.cloud_attenuation_db(
        lat_deg,
        lon_deg,
        elevation_deg,
        freq_ghz,
        p,
        lred_kgm2,
    ))
}

/// Computes wet-term radio refractivity from ITU-R [P.453](https://www.itu.int/rec/R-REC-P.453).
///
/// Wet-term refractivity is the water-vapour part of atmospheric radio
/// refractivity. It is used by scintillation calculations and by atmospheric
/// refractive-index estimates.
///
/// # Parameters
///
/// - `e_hpa`: water-vapour partial pressure in hPa. Must be non-negative.
/// - `temp_c`: air temperature in degrees Celsius. Must be above absolute
///   zero.
///
/// # Returns
///
/// Wet-term radio refractivity in N-units.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let n_wet = itu_rs::wet_term_radio_refractivity(12.0, 15.0)?;
///
/// assert!(n_wet >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn wet_term_radio_refractivity(e_hpa: f64, temp_c: f64) -> std::result::Result<f64, ItuError> {
    validate_nonnegative("e_hpa", e_hpa)
        .and_then(|_| validate_finite("temp_c", temp_c))
        .and_then(|_| {
            if temp_c <= -273.15 {
                Err("temp_c must be > -273.15".to_string())
            } else {
                Ok(())
            }
        })
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .wet_term_radio_refractivity(e_hpa, temp_c))
}

/// Computes radio refractive index from ITU-R [P.453](https://www.itu.int/rec/R-REC-P.453).
///
/// This converts dry-air pressure, water-vapour pressure, and temperature into
/// the dimensionless radio refractive index `n`. The value is close to `1.0`
/// in the lower atmosphere.
///
/// # Parameters
///
/// - `pd_hpa`: dry-air partial pressure in hPa. Must be non-negative.
/// - `e_hpa`: water-vapour partial pressure in hPa. Must be non-negative.
/// - `temp_k`: absolute air temperature in kelvin. Must be positive.
///
/// # Returns
///
/// Dimensionless radio refractive index.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let n = itu_rs::radio_refractive_index(1000.0, 12.0, 288.15)?;
///
/// assert!(n > 1.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn radio_refractive_index(
    pd_hpa: f64,
    e_hpa: f64,
    temp_k: f64,
) -> std::result::Result<f64, ItuError> {
    validate_nonnegative("pd_hpa", pd_hpa)
        .and_then(|_| validate_nonnegative("e_hpa", e_hpa))
        .and_then(|_| validate_positive("temp_k", temp_k))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .radio_refractive_index(pd_hpa, e_hpa, temp_k))
}

/// Computes water-vapour pressure from ITU-R [P.453](https://www.itu.int/rec/R-REC-P.453).
///
/// This converts air temperature, total pressure, and relative humidity into
/// water-vapour partial pressure. It is useful when supplying local
/// scintillation meteorology instead of using map-derived wet refractivity.
///
/// # Parameters
///
/// - `temp_c`: air temperature in degrees Celsius. Must be above absolute
///   zero.
/// - `pressure_hpa`: total atmospheric pressure in hPa. Must be positive.
/// - `humidity_percent`: relative humidity in percent, in `[0, 100]`.
///
/// # Returns
///
/// Water-vapour partial pressure in hPa.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let e_hpa = itu_rs::water_vapour_pressure_hpa(15.0, 1000.0, 60.0)?;
///
/// assert!(e_hpa >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn water_vapour_pressure_hpa(
    temp_c: f64,
    pressure_hpa: f64,
    humidity_percent: f64,
) -> std::result::Result<f64, ItuError> {
    validate_finite("temp_c", temp_c)
        .and_then(|_| validate_positive("pressure_hpa", pressure_hpa))
        .and_then(|_| validate_finite("humidity_percent", humidity_percent))
        .and_then(|_| {
            if temp_c <= -273.15 {
                Err("temp_c must be > -273.15".to_string())
            } else if !(0.0..=100.0).contains(&humidity_percent) {
                Err("humidity_percent must be in [0, 100]".to_string())
            } else {
                Ok(())
            }
        })
        .map_err(ItuError::from)?;
    Ok(model().map_err(ItuError::from)?.water_vapour_pressure_hpa(
        temp_c,
        pressure_hpa,
        humidity_percent,
    ))
}

/// Looks up wet-term radio refractivity maps from ITU-R [P.453](https://www.itu.int/rec/R-REC-P.453).
///
/// This map lookup provides wet-term radio refractivity exceeded for a requested
/// time percentage. Scintillation calculations use it when local humidity,
/// temperature, and pressure are not supplied.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `p`: percentage of time exceeded. Must be positive.
///
/// # Returns
///
/// Wet-term radio refractivity in N-units.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/453/v13_lat_n.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let n_wet = itu_rs::map_wet_term_radio_refractivity(45.4215, -75.6972, 50.0)?;
///
/// assert!(n_wet.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn map_wet_term_radio_refractivity(
    lat_deg: f64,
    lon_deg: f64,
    p: f64,
) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg)
        .and_then(|_| validate_p(p))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .map_wet_term_radio_refractivity(lat_deg, lon_deg, p))
}

/// Computes dry-air specific attenuation from ITU-R [P.676](https://www.itu.int/rec/R-REC-P.676) in dB/km.
///
/// This is the oxygen and dry-air part of exact gaseous attenuation for a local
/// atmospheric state. It does not include water-vapour line attenuation.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `pressure_hpa`: atmospheric pressure in hPa. Must be positive.
/// - `rho_gm3`: water-vapour density in g/m^3. Must be non-negative because it
///   contributes to line broadening even in the dry-air term.
/// - `temp_k`: absolute air temperature in kelvin. Must be positive.
///
/// # Returns
///
/// Dry-air specific attenuation in dB/km.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/676/v13_lines_oxygen.txt")
/// #             .exists()
/// # }
/// # if data_available() {
/// let gamma0 = itu_rs::gamma0_exact_db_per_km(12.0, 1008.0, 7.5, 289.2)?;
///
/// assert!(gamma0 >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gamma0_exact_db_per_km(
    freq_ghz: f64,
    pressure_hpa: f64,
    rho_gm3: f64,
    temp_k: f64,
) -> std::result::Result<f64, ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_positive("pressure_hpa", pressure_hpa))
        .and_then(|_| validate_nonnegative("rho_gm3", rho_gm3))
        .and_then(|_| validate_positive("temp_k", temp_k))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .gamma0_exact_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k))
}

/// Computes water-vapour specific attenuation from ITU-R [P.676](https://www.itu.int/rec/R-REC-P.676) in dB/km.
///
/// This is the water-vapour line and continuum part of exact gaseous
/// attenuation for a local atmospheric state.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `pressure_hpa`: atmospheric pressure in hPa. Must be positive.
/// - `rho_gm3`: water-vapour density in g/m^3. Must be non-negative.
/// - `temp_k`: absolute air temperature in kelvin. Must be positive.
///
/// # Returns
///
/// Water-vapour specific attenuation in dB/km.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/676/v13_lines_water_vapour.txt")
/// #             .exists()
/// # }
/// # if data_available() {
/// let gammaw = itu_rs::gammaw_exact_db_per_km(12.0, 1008.0, 7.5, 289.2)?;
///
/// assert!(gammaw >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gammaw_exact_db_per_km(
    freq_ghz: f64,
    pressure_hpa: f64,
    rho_gm3: f64,
    temp_k: f64,
) -> std::result::Result<f64, ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_positive("pressure_hpa", pressure_hpa))
        .and_then(|_| validate_nonnegative("rho_gm3", rho_gm3))
        .and_then(|_| validate_positive("temp_k", temp_k))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .gammaw_exact_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k))
}

/// Computes total specific gaseous attenuation from ITU-R [P.676](https://www.itu.int/rec/R-REC-P.676) in dB/km.
///
/// This is the sum of dry-air and water-vapour specific attenuation for a local
/// atmospheric state.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `pressure_hpa`: atmospheric pressure in hPa. Must be positive.
/// - `rho_gm3`: water-vapour density in g/m^3. Must be non-negative.
/// - `temp_k`: absolute air temperature in kelvin. Must be positive.
///
/// # Returns
///
/// Total gaseous specific attenuation in dB/km.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/676/v13_lines_oxygen.txt")
/// #             .exists()
/// # }
/// # if data_available() {
/// let gamma = itu_rs::gamma_exact_db_per_km(12.0, 1008.0, 7.5, 289.2)?;
///
/// assert!(gamma >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gamma_exact_db_per_km(
    freq_ghz: f64,
    pressure_hpa: f64,
    rho_gm3: f64,
    temp_k: f64,
) -> std::result::Result<f64, ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_positive("pressure_hpa", pressure_hpa))
        .and_then(|_| validate_nonnegative("rho_gm3", rho_gm3))
        .and_then(|_| validate_positive("temp_k", temp_k))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .gamma_exact_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k))
}

/// Computes [P.676](https://www.itu.int/rec/R-REC-P.676) equivalent heights for dry air and water vapour.
///
/// Equivalent heights convert local specific gaseous attenuation into an
/// approximate slant-path attenuation. The returned dry-air and water-vapour
/// heights are used by the approximate [P.676](https://www.itu.int/rec/R-REC-P.676) path calculation.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `pressure_hpa`: atmospheric pressure in hPa. Must be positive.
/// - `rho_gm3`: surface water-vapour density in g/m^3. Must be non-negative.
/// - `temp_k`: absolute air temperature in kelvin. Must be positive.
///
/// # Returns
///
/// `(h0_km, hw_km)`, the dry-air and water-vapour equivalent heights in
/// kilometres.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/676/v13_h0_coefficients.txt")
/// #             .exists()
/// # }
/// # if data_available() {
/// let (h0_km, hw_km) = itu_rs::slant_inclined_path_equivalent_height_km(
///     12.0, 1008.0, 7.5, 289.2,
/// )?;
///
/// assert!(h0_km.is_finite());
/// assert!(hw_km.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn slant_inclined_path_equivalent_height_km(
    freq_ghz: f64,
    pressure_hpa: f64,
    rho_gm3: f64,
    temp_k: f64,
) -> std::result::Result<(f64, f64), ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_positive("pressure_hpa", pressure_hpa))
        .and_then(|_| validate_nonnegative("rho_gm3", rho_gm3))
        .and_then(|_| validate_positive("temp_k", temp_k))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .slant_inclined_path_equivalent_height_v13(freq_ghz, pressure_hpa, rho_gm3, temp_k))
}

/// Computes [P.676](https://www.itu.int/rec/R-REC-P.676) zenith water-vapour attenuation.
///
/// This estimates the water-vapour attenuation for a vertical path above a
/// station. It is one input to the approximate gaseous slant-path attenuation.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `v_t_kgm2`: total columnar water-vapour content in kg/m^2. Must be
///   non-negative.
/// - `h_km`: station altitude in kilometres. Must be non-negative.
///
/// # Returns
///
/// Zenith water-vapour attenuation in dB.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/676/v13_h0_coefficients.txt")
/// #             .exists()
/// # }
/// # if data_available() {
/// let zenith_db = itu_rs::zenith_water_vapour_attenuation_db(12.0, 22.5, 0.4)?;
///
/// assert!(zenith_db >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn zenith_water_vapour_attenuation_db(
    freq_ghz: f64,
    v_t_kgm2: f64,
    h_km: f64,
) -> std::result::Result<f64, ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_nonnegative("v_t_kgm2", v_t_kgm2))
        .and_then(|_| validate_nonnegative("h_km", h_km))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .zenith_water_vapour_attenuation_db(freq_ghz, v_t_kgm2, h_km))
}

/// Computes gaseous attenuation on an Earth-space slant path from ITU-R [P.676](https://www.itu.int/rec/R-REC-P.676).
///
/// Gaseous attenuation is absorption by oxygen and water vapour along the
/// Earth-space path. Use `exact = true` for the layer-integrated model or
/// `exact = false` for the faster approximate model based on equivalent
/// heights and zenith water-vapour attenuation.
///
/// # Parameters
///
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: path elevation angle above the horizon in degrees, in
///   `(0, 90)`.
/// - `rho_gm3`: surface water-vapour density in g/m^3. Must be non-negative.
/// - `pressure_hpa`: surface atmospheric pressure in hPa. Must be positive.
/// - `temp_k`: surface air temperature in kelvin. Must be positive.
/// - `v_t_kgm2`: total columnar water-vapour content in kg/m^2. Must be
///   non-negative.
/// - `h_km`: station altitude in kilometres. Must be non-negative.
/// - `exact`: when `true`, use exact layer integration; when `false`, use the
///   approximate slant-path method.
///
/// # Returns
///
/// Gaseous slant-path attenuation in dB.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/676/v13_lines_oxygen.txt")
/// #             .exists()
/// # }
/// # if data_available() {
/// let gas_db = itu_rs::gaseous_attenuation_slant_path_db(
///     12.0, 30.0, 7.5, 1008.0, 289.2, 22.5, 0.4, false,
/// )?;
///
/// assert!(gas_db >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gaseous_attenuation_slant_path_db(
    freq_ghz: f64,
    elevation_deg: f64,
    rho_gm3: f64,
    pressure_hpa: f64,
    temp_k: f64,
    v_t_kgm2: f64,
    h_km: f64,
    exact: bool,
) -> std::result::Result<f64, ItuError> {
    validate_positive("freq_ghz", freq_ghz)
        .and_then(|_| validate_elevation_deg(elevation_deg))
        .and_then(|_| validate_nonnegative("rho_gm3", rho_gm3))
        .and_then(|_| validate_positive("pressure_hpa", pressure_hpa))
        .and_then(|_| validate_positive("temp_k", temp_k))
        .and_then(|_| validate_nonnegative("v_t_kgm2", v_t_kgm2))
        .and_then(|_| validate_nonnegative("h_km", h_km))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .gaseous_attenuation_slant_path_v13(
            freq_ghz,
            elevation_deg,
            rho_gm3,
            pressure_hpa,
            temp_k,
            v_t_kgm2,
            h_km,
            exact,
        ))
}

/// Computes rain attenuation from ITU-R [P.618](https://www.itu.int/rec/R-REC-P.618).
///
/// Rain attenuation is the fade contribution from precipitation along the
/// Earth-space path. The model combines local rain rate, rain height, station
/// height, path geometry, frequency, time percentage, and polarization.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: path elevation angle in degrees, in `(0, 90)`.
/// - `hs_km`: station altitude above mean sea level in kilometres. Any finite
///   value is accepted.
/// - `p`: percentage of time exceeded. Must be positive.
/// - `r001_mmh`: optional rainfall rate exceeded for 0.01% of an average year
///   in mm/h. When `None`, [P.837](https://www.itu.int/rec/R-REC-P.837) data are used.
/// - `tau_deg`: polarization tilt angle in degrees, in `[0, 90]`.
/// - `l_s_km`: optional slant-path length through rain in kilometres. When
///   `None`, [P.618](https://www.itu.int/rec/R-REC-P.618) derives it from rain height and elevation.
///
/// # Returns
///
/// Rain attenuation in dB for the requested time percentage.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/837/v7_lat_r001.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let rain_db = itu_rs::rain_attenuation_db(
///     45.4215, -75.6972, 12.0, 30.0, 0.4, 1.0, None, 45.0, None,
/// )?;
///
/// assert!(rain_db >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn rain_attenuation_db(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    hs_km: f64,
    p: f64,
    r001_mmh: Option<f64>,
    tau_deg: f64,
    l_s_km: Option<f64>,
) -> std::result::Result<f64, ItuError> {
    validate_lat_lon(lat_deg, lon_deg)
        .and_then(|_| validate_positive("freq_ghz", freq_ghz))
        .and_then(|_| validate_elevation_deg(elevation_deg))
        .and_then(|_| validate_finite("hs_km", hs_km))
        .and_then(|_| validate_p(p))
        .and_then(|_| validate_optional_nonnegative("r001_mmh", r001_mmh))
        .and_then(|_| validate_tau_deg(tau_deg))
        .and_then(|_| validate_optional_positive("l_s_km", l_s_km))
        .map_err(ItuError::from)?;
    Ok(model().map_err(ItuError::from)?.rain_attenuation_db(
        lat_deg,
        lon_deg,
        freq_ghz,
        elevation_deg,
        hs_km,
        p,
        r001_mmh,
        tau_deg,
        l_s_km,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_scintillation_inputs(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    dish_m: f64,
    eta: f64,
    temp_c: Option<f64>,
    humidity_percent: Option<f64>,
    pressure_hpa: Option<f64>,
    h_l_m: f64,
) -> Result<(), String> {
    validate_lat_lon(lat_deg, lon_deg)
        .and_then(|_| validate_positive("freq_ghz", freq_ghz))
        .and_then(|_| validate_elevation_deg(elevation_deg))
        .and_then(|_| validate_positive("dish_m", dish_m))
        .and_then(|_| validate_positive("h_l_m", h_l_m))
        .and_then(|_| validate_finite("eta", eta))?;
    if eta > 1.0 {
        return Err("eta must be in (0, 1]".to_string());
    }

    match (temp_c, humidity_percent, pressure_hpa) {
        (None, None, None) => Ok(()),
        (Some(t), Some(h), Some(p_hpa)) => validate_finite("temp_c", t)
            .and_then(|_| {
                if t <= -273.15 {
                    Err("temp_c must be > -273.15".to_string())
                } else {
                    Ok(())
                }
            })
            .and_then(|_| validate_finite("humidity_percent", h))
            .and_then(|_| {
                if !(0.0..=100.0).contains(&h) {
                    Err("humidity_percent must be in [0, 100]".to_string())
                } else {
                    Ok(())
                }
            })
            .and_then(|_| validate_positive("pressure_hpa", p_hpa)),
        _ => {
            Err("temp_c, humidity_percent, and pressure_hpa must be supplied together".to_string())
        }
    }
}

/// Computes the [P.618](https://www.itu.int/rec/R-REC-P.618) scintillation standard deviation in dB.
///
/// This is the standard deviation of short-term tropospheric scintillation
/// amplitude fluctuations before converting to a fade depth for a requested
/// time percentage.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: path elevation angle in degrees, in `(0, 90)`.
/// - `dish_m`: physical antenna diameter in metres. Must be positive.
/// - `eta`: aperture efficiency factor, in `(0, 1]`.
/// - `temp_c`, `humidity_percent`, `pressure_hpa`: optional local meteorology.
///   Supply all three together to compute wet refractivity from local
///   conditions, or supply all as `None` to use [P.453](https://www.itu.int/rec/R-REC-P.453) map data.
/// - `h_l_m`: turbulent layer height in metres. Must be positive.
///
/// # Returns
///
/// Scintillation standard deviation in dB.
///
/// # Errors
///
/// Returns an error if inputs fail validation, if only part of the optional
/// meteorology set is supplied, or if model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/453/v13_lat_n.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let sigma_db = itu_rs::scintillation_sigma_db(
///     45.4215, -75.6972, 12.0, 30.0, 1.2, 0.5, None, None, None, 1000.0,
/// )?;
///
/// assert!(sigma_db >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn scintillation_sigma_db(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    dish_m: f64,
    eta: f64,
    temp_c: Option<f64>,
    humidity_percent: Option<f64>,
    pressure_hpa: Option<f64>,
    h_l_m: f64,
) -> std::result::Result<f64, ItuError> {
    validate_scintillation_inputs(
        lat_deg,
        lon_deg,
        freq_ghz,
        elevation_deg,
        dish_m,
        eta,
        temp_c,
        humidity_percent,
        pressure_hpa,
        h_l_m,
    )
    .map_err(ItuError::from)?;
    Ok(model().map_err(ItuError::from)?.scintillation_sigma_db(
        lat_deg,
        lon_deg,
        freq_ghz,
        elevation_deg,
        dish_m,
        eta,
        temp_c,
        humidity_percent,
        pressure_hpa,
        h_l_m,
    ))
}

/// Computes scintillation attenuation from ITU-R [P.618](https://www.itu.int/rec/R-REC-P.618).
///
/// This converts the scintillation standard deviation into an attenuation
/// exceeded for the requested time percentage. It represents the fade
/// contribution from tropospheric turbulence.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: path elevation angle in degrees, in `(0, 90)`.
/// - `p`: percentage of time exceeded. Must be positive.
/// - `dish_m`: physical antenna diameter in metres. Must be positive.
/// - `eta`: aperture efficiency factor, in `(0, 1]`.
/// - `temp_c`, `humidity_percent`, `pressure_hpa`: optional local meteorology.
///   Supply all three together, or supply all as `None` to use [P.453](https://www.itu.int/rec/R-REC-P.453) map data.
/// - `h_l_m`: turbulent layer height in metres. Must be positive.
///
/// # Returns
///
/// Scintillation attenuation in dB.
///
/// # Errors
///
/// Returns an error if inputs fail validation, if only part of the optional
/// meteorology set is supplied, or if model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/453/v13_lat_n.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let scint_db = itu_rs::scintillation_attenuation_db(
///     45.4215, -75.6972, 12.0, 30.0, 1.0, 1.2, 0.5,
///     None, None, None, 1000.0,
/// )?;
///
/// assert!(scint_db >= 0.0);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn scintillation_attenuation_db(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    p: f64,
    dish_m: f64,
    eta: f64,
    temp_c: Option<f64>,
    humidity_percent: Option<f64>,
    pressure_hpa: Option<f64>,
    h_l_m: f64,
) -> std::result::Result<f64, ItuError> {
    validate_p(p)
        .and_then(|_| {
            validate_scintillation_inputs(
                lat_deg,
                lon_deg,
                freq_ghz,
                elevation_deg,
                dish_m,
                eta,
                temp_c,
                humidity_percent,
                pressure_hpa,
                h_l_m,
            )
        })
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .scintillation_attenuation_db(
            lat_deg,
            lon_deg,
            freq_ghz,
            elevation_deg,
            p,
            dish_m,
            eta,
            temp_c,
            humidity_percent,
            pressure_hpa,
            h_l_m,
        ))
}

/// Computes gas-only atmospheric attenuation for one elevation angle.
///
/// This uses the same default environmental lookups as the supported
/// slant-path attenuation path, but disables rain, cloud, and scintillation
/// contributions.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: path elevation angle above the horizon in degrees, in
///   `(0, 90)`.
/// - `p`: percentage of time exceeded. Must be positive.
/// - `d_m`: physical antenna diameter in metres. It is accepted for API
///   compatibility with the full slant-path call and must be positive, but
///   gas-only attenuation does not use antenna size physically.
///
/// # Returns
///
/// Gaseous attenuation in dB.
///
/// # Errors
///
/// Returns an error if inputs fail validation or model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let gas_db = itu_rs::gas_attenuation_default(
///     45.4215, -75.6972, 12.0, 30.0, 0.1, 1.2,
/// )?;
///
/// assert!(gas_db.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gas_attenuation_default(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    p: f64,
    d_m: f64,
) -> std::result::Result<f64, ItuError> {
    validate_common_inputs(lat_deg, lon_deg, freq_ghz, p, d_m)
        .and_then(|_| validate_elevation_deg(elevation_deg))
        .map_err(ItuError::from)?;
    Ok(model()
        .map_err(ItuError::from)?
        .atmospheric_attenuation_default_gas_only(
            lat_deg,
            lon_deg,
            freq_ghz,
            elevation_deg,
            p,
            d_m,
        ))
}

/// Computes gas-only atmospheric attenuation for multiple elevation angles.
///
/// Elevation values are validated exactly. Use only values in the open interval
/// `(0, 90)`.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: slice of path elevation angles in degrees. Every value
///   must be in `(0, 90)`.
/// - `p`: percentage of time exceeded. Must be positive.
/// - `d_m`: physical antenna diameter in metres. It is validated for
///   compatibility with the full slant-path API but is not used by gas-only
///   attenuation.
///
/// # Returns
///
/// One gaseous attenuation value in dB for each supplied elevation angle, in
/// the same order as the input slice.
///
/// # Errors
///
/// Returns an error if any input fails validation or model data cannot be
/// loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let elevations = [5.0, 30.0, 89.0];
/// let gas = itu_rs::gas_attenuation_default_many(
///     45.4215, -75.6972, 12.0, &elevations, 0.1, 1.2,
/// )?;
///
/// assert_eq!(gas.len(), elevations.len());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gas_attenuation_default_many(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: &[f64],
    p: f64,
    d_m: f64,
) -> std::result::Result<Vec<f64>, ItuError> {
    gas_attenuation_default_many_checked(lat_deg, lon_deg, freq_ghz, elevation_deg, p, d_m)
}

/// Computes gas-only atmospheric attenuation for multiple elevation angles.
///
/// This is retained as a compatibility alias for callers that want the name to
/// emphasize strict validation.
///
/// # Parameters
///
/// - `lat_deg`: station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: slice of path elevation angles in degrees. Every value
///   must be in `(0, 90)`.
/// - `p`: percentage of time exceeded. Must be positive.
/// - `d_m`: physical antenna diameter in metres. Must be positive.
///
/// # Returns
///
/// One gaseous attenuation value in dB for each supplied elevation angle.
///
/// # Errors
///
/// Returns an error if any input fails validation or model data cannot be
/// loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// let elevations = [10.0, 45.0];
/// let gas = itu_rs::gas_attenuation_default_many_checked(
///     45.4215, -75.6972, 12.0, &elevations, 0.1, 1.2,
/// )?;
///
/// assert_eq!(gas.len(), elevations.len());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gas_attenuation_default_many_checked(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: &[f64],
    p: f64,
    d_m: f64,
) -> std::result::Result<Vec<f64>, ItuError> {
    validate_common_inputs(lat_deg, lon_deg, freq_ghz, p, d_m).map_err(ItuError::from)?;
    let mut out = Vec::with_capacity(elevation_deg.len());
    let model = model().map_err(ItuError::from)?;
    for &el in elevation_deg {
        validate_elevation_deg(el).map_err(ItuError::from)?;
        out.push(
            model.atmospheric_attenuation_default_gas_only(lat_deg, lon_deg, freq_ghz, el, p, d_m),
        );
    }
    Ok(out)
}

/// Computes total atmospheric attenuation for one Earth-space slant path.
///
/// The returned [`SlantPathContributions`] contains gas, cloud, rain,
/// scintillation, and total attenuation in dB. Use [`SlantPathOptions`] to
/// override environmental inputs, select exact gaseous attenuation, or disable
/// individual components.
///
/// # Parameters
///
/// - `lat_deg`: Earth station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: Earth station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: link elevation angle above the local horizon in degrees,
///   in `(0, 90)`.
/// - `p`: percentage of time attenuation is exceeded. Must be positive; for
///   example, `0.1` means 0.1% of an average year.
/// - `d_m`: physical antenna diameter in metres. It affects scintillation fade
///   through aperture averaging and must be positive.
/// - `options`: environmental overrides and contribution switches. Defaults
///   look up missing environmental values from ITU-R maps and include all
///   supported components.
///
/// # Returns
///
/// A [`SlantPathContributions`] value containing gas, cloud, rain,
/// scintillation, and combined total attenuation in dB.
///
/// # Errors
///
/// Returns an error if inputs or options fail validation, or if required model
/// data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// use itu_rs::{atmospheric_attenuation_slant_path, SlantPathOptions};
///
/// let options = SlantPathOptions {
///     exact: true,
///     ..SlantPathOptions::default()
/// };
///
/// let attenuation = atmospheric_attenuation_slant_path(
///     10.0, 20.0, 18.0, 17.5, 0.7, 0.8, options,
/// )?;
///
/// assert!(attenuation.total_db.is_finite());
/// assert!(attenuation.gas_db.is_finite());
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn atmospheric_attenuation_slant_path(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    p: f64,
    d_m: f64,
    options: SlantPathOptions,
) -> std::result::Result<SlantPathContributions, ItuError> {
    rust_itur_slant_path_scalar(lat_deg, lon_deg, freq_ghz, elevation_deg, p, d_m, options)
        .map_err(ItuError::from)
}

/// Computes atmospheric attenuation for multiple elevation angles.
///
/// This is the preferred API when sweeping elevation angles for a fixed site,
/// frequency, time percentage, antenna diameter, and option set.
///
/// # Parameters
///
/// - `lat_deg`: Earth station geodetic latitude in degrees, in `[-90, 90]`.
/// - `lon_deg`: Earth station geodetic longitude in degrees.
/// - `freq_ghz`: carrier frequency in GHz. Must be positive.
/// - `elevation_deg`: slice of link elevation angles in degrees. Every value
///   must be in `(0, 90)`.
/// - `p`: percentage of time attenuation is exceeded. Must be positive.
/// - `d_m`: physical antenna diameter in metres. Must be positive.
/// - `options`: environmental overrides and contribution switches applied to
///   every elevation angle.
///
/// # Returns
///
/// One [`SlantPathContributions`] value per elevation angle, in input order.
///
/// # Errors
///
/// Returns an error if common inputs, options, or any elevation angle fail
/// validation, or if required model data cannot be loaded.
///
/// # Example
///
/// ```
/// # fn data_available() -> bool {
/// #     cfg!(feature = "data")
/// #         || std::env::var_os("ITU_RS_DATA_DIR").is_some()
/// #         || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
/// #             .join("data/1511/v2_lat.npz")
/// #             .exists()
/// # }
/// # if data_available() {
/// use itu_rs::{atmospheric_attenuation_slant_path_many, SlantPathOptions};
///
/// let elevations = [5.0, 17.5, 45.0, 89.0];
/// let attenuation = atmospheric_attenuation_slant_path_many(
///     45.4215,
///     -75.6972,
///     12.0,
///     &elevations,
///     0.1,
///     1.2,
///     SlantPathOptions::default(),
/// )?;
///
/// assert_eq!(attenuation.len(), elevations.len());
/// assert!(attenuation.iter().all(|item| item.total_db.is_finite()));
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn atmospheric_attenuation_slant_path_many(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: &[f64],
    p: f64,
    d_m: f64,
    options: SlantPathOptions,
) -> std::result::Result<Vec<SlantPathContributions>, ItuError> {
    validate_common_inputs(lat_deg, lon_deg, freq_ghz, p, d_m)
        .and_then(|_| validate_options(options))
        .map_err(ItuError::from)?;

    elevation_deg
        .iter()
        .map(|&el| rust_itur_slant_path_scalar(lat_deg, lon_deg, freq_ghz, el, p, d_m, options))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(ItuError::from)
}
