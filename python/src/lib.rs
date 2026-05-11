#![allow(clippy::too_many_arguments)]

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use itu_rs_core::{
    GasPathMode as CoreGasPathMode, HydrometeorType as CoreHydrometeorType,
    SlantPathContributions as CoreSlantPathContributions, SlantPathOptions as CoreSlantPathOptions,
};
use ndarray::Array2;

create_exception!(_native, PyItuError, PyException);

fn into_py_err(err: itu_rs_core::ItuError) -> PyErr {
    PyItuError::new_err(err.to_string())
}

macro_rules! py_result_fn {
    ($(#[$meta:meta])* fn $name:ident($($arg:ident: $typ:ty),* $(,)?) -> $ret:ty => $target:path) => {
        #[pyfunction]
        $(#[$meta])*
        fn $name($($arg: $typ),*) -> PyResult<$ret> {
            $target($($arg),*).map_err(into_py_err)
        }
    };
}

macro_rules! add_py_functions {
    ($module:ident, [$($name:ident),* $(,)?]) => {
        $(
            $module.add_function(wrap_pyfunction!($name, $module)?)?;
        )*
    };
}

#[pyclass(name = "SlantPathOptions")]
#[derive(Clone, Debug)]
struct PySlantPathOptions {
    #[pyo3(get, set)]
    hs_km: Option<f64>,
    #[pyo3(get, set)]
    rho_gm3: Option<f64>,
    #[pyo3(get, set)]
    r001_mmh: Option<f64>,
    #[pyo3(get, set)]
    eta: f64,
    #[pyo3(get, set)]
    t: Option<f64>,
    #[pyo3(get, set)]
    h_percent: Option<f64>,
    #[pyo3(get, set)]
    pressure_hpa: Option<f64>,
    #[pyo3(get, set)]
    h_l_m: f64,
    #[pyo3(get, set)]
    l_s_km: Option<f64>,
    #[pyo3(get, set)]
    tau_deg: f64,
    #[pyo3(get, set)]
    v_t_kgm2: Option<f64>,
    #[pyo3(get, set)]
    exact: bool,
    #[pyo3(get, set)]
    include_rain: bool,
    #[pyo3(get, set)]
    include_gas: bool,
    #[pyo3(get, set)]
    include_scintillation: bool,
    #[pyo3(get, set)]
    include_clouds: bool,
}

#[pymethods]
impl PySlantPathOptions {
    #[new]
    #[pyo3(signature = (
        hs_km=None,
        rho_gm3=None,
        r001_mmh=None,
        eta=0.5,
        t=None,
        h_percent=None,
        pressure_hpa=None,
        h_l_m=1000.0,
        l_s_km=None,
        tau_deg=45.0,
        v_t_kgm2=None,
        exact=false,
        include_rain=true,
        include_gas=true,
        include_scintillation=true,
        include_clouds=true
    ))]
    fn new(
        hs_km: Option<f64>,
        rho_gm3: Option<f64>,
        r001_mmh: Option<f64>,
        eta: f64,
        t: Option<f64>,
        h_percent: Option<f64>,
        pressure_hpa: Option<f64>,
        h_l_m: f64,
        l_s_km: Option<f64>,
        tau_deg: f64,
        v_t_kgm2: Option<f64>,
        exact: bool,
        include_rain: bool,
        include_gas: bool,
        include_scintillation: bool,
        include_clouds: bool,
    ) -> Self {
        Self {
            hs_km,
            rho_gm3,
            r001_mmh,
            eta,
            t,
            h_percent,
            pressure_hpa,
            h_l_m,
            l_s_km,
            tau_deg,
            v_t_kgm2,
            exact,
            include_rain,
            include_gas,
            include_scintillation,
            include_clouds,
        }
    }

    #[staticmethod]
    fn default() -> Self {
        CoreSlantPathOptions::default().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "SlantPathOptions(exact={}, include_rain={}, include_gas={}, include_scintillation={}, include_clouds={})",
            self.exact,
            self.include_rain,
            self.include_gas,
            self.include_scintillation,
            self.include_clouds
        )
    }
}

impl PySlantPathOptions {
    fn to_core(&self) -> CoreSlantPathOptions {
        CoreSlantPathOptions {
            hs_km: self.hs_km,
            rho_gm3: self.rho_gm3,
            r001_mmh: self.r001_mmh,
            eta: self.eta,
            t: self.t,
            h_percent: self.h_percent,
            pressure_hpa: self.pressure_hpa,
            h_l_m: self.h_l_m,
            l_s_km: self.l_s_km,
            tau_deg: self.tau_deg,
            v_t_kgm2: self.v_t_kgm2,
            exact: self.exact,
            include_rain: self.include_rain,
            include_gas: self.include_gas,
            include_scintillation: self.include_scintillation,
            include_clouds: self.include_clouds,
        }
    }
}

impl From<CoreSlantPathOptions> for PySlantPathOptions {
    fn from(options: CoreSlantPathOptions) -> Self {
        Self {
            hs_km: options.hs_km,
            rho_gm3: options.rho_gm3,
            r001_mmh: options.r001_mmh,
            eta: options.eta,
            t: options.t,
            h_percent: options.h_percent,
            pressure_hpa: options.pressure_hpa,
            h_l_m: options.h_l_m,
            l_s_km: options.l_s_km,
            tau_deg: options.tau_deg,
            v_t_kgm2: options.v_t_kgm2,
            exact: options.exact,
            include_rain: options.include_rain,
            include_gas: options.include_gas,
            include_scintillation: options.include_scintillation,
            include_clouds: options.include_clouds,
        }
    }
}

#[pyclass(name = "SlantPathContributions", frozen)]
#[derive(Clone, Copy, Debug)]
struct PySlantPathContributions {
    #[pyo3(get)]
    gas_db: f64,
    #[pyo3(get)]
    cloud_db: f64,
    #[pyo3(get)]
    rain_db: f64,
    #[pyo3(get)]
    scintillation_db: f64,
    #[pyo3(get)]
    total_db: f64,
}

#[pymethods]
impl PySlantPathContributions {
    fn __repr__(&self) -> String {
        format!(
            "SlantPathContributions(gas_db={}, cloud_db={}, rain_db={}, scintillation_db={}, total_db={})",
            self.gas_db, self.cloud_db, self.rain_db, self.scintillation_db, self.total_db
        )
    }
}

impl From<CoreSlantPathContributions> for PySlantPathContributions {
    fn from(value: CoreSlantPathContributions) -> Self {
        Self {
            gas_db: value.gas_db,
            cloud_db: value.cloud_db,
            rain_db: value.rain_db,
            scintillation_db: value.scintillation_db,
            total_db: value.total_db,
        }
    }
}

#[pyclass(name = "HydrometeorType", frozen)]
#[derive(Clone, Copy, Debug)]
struct PyHydrometeorType {
    inner: CoreHydrometeorType,
}

#[pymethods]
impl PyHydrometeorType {
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_hydrometeor_str(value)?,
        })
    }

    #[staticmethod]
    fn water() -> Self {
        Self {
            inner: CoreHydrometeorType::Water,
        }
    }

    #[staticmethod]
    fn ice() -> Self {
        Self {
            inner: CoreHydrometeorType::Ice,
        }
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CoreHydrometeorType::Water => "water",
            CoreHydrometeorType::Ice => "ice",
        }
    }

    fn __repr__(&self) -> String {
        format!("HydrometeorType({:?})", self.name())
    }
}

fn parse_hydrometeor_str(value: &str) -> PyResult<CoreHydrometeorType> {
    match value.to_ascii_lowercase().as_str() {
        "water" => Ok(CoreHydrometeorType::Water),
        "ice" => Ok(CoreHydrometeorType::Ice),
        _ => Err(PyValueError::new_err(
            "hydrometeor must be 'water', 'ice', or a HydrometeorType instance",
        )),
    }
}

fn parse_hydrometeor(value: &Bound<'_, PyAny>) -> PyResult<CoreHydrometeorType> {
    if let Ok(hydrometeor) = value.extract::<PyRef<'_, PyHydrometeorType>>() {
        return Ok(hydrometeor.inner);
    }
    if let Ok(text) = value.extract::<String>() {
        return parse_hydrometeor_str(&text);
    }
    Err(PyTypeError::new_err(
        "hydrometeor must be 'water', 'ice', or a HydrometeorType instance",
    ))
}

fn parse_gas_path_mode(value: &Bound<'_, PyAny>) -> PyResult<CoreGasPathMode> {
    if let Ok(exact) = value.extract::<bool>() {
        return Ok(if exact {
            CoreGasPathMode::Exact
        } else {
            CoreGasPathMode::Approximate
        });
    }
    if let Ok(text) = value.extract::<String>() {
        return match text.to_ascii_lowercase().as_str() {
            "approx" | "approximate" => Ok(CoreGasPathMode::Approximate),
            "exact" => Ok(CoreGasPathMode::Exact),
            _ => Err(PyValueError::new_err(
                "mode must be 'approximate', 'exact', false, or true",
            )),
        };
    }
    Err(PyTypeError::new_err(
        "mode must be 'approximate', 'exact', false, or true",
    ))
}

fn array2_from_rows(rows: Vec<Vec<f64>>, name: &str) -> PyResult<Array2<f64>> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    if nrows == 0 || ncols == 0 {
        return Err(PyValueError::new_err(format!("{name} must not be empty")));
    }
    if rows.iter().any(|row| row.len() != ncols) {
        return Err(PyValueError::new_err(format!(
            "{name} rows must all have the same length"
        )));
    }
    Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect())
        .map_err(|err| PyValueError::new_err(format!("invalid {name}: {err}")))
}

#[pyfunction]
fn is_regular_grid(lat_grid: Vec<Vec<f64>>, lon_grid: Vec<Vec<f64>>) -> PyResult<bool> {
    let lat_grid = array2_from_rows(lat_grid, "lat_grid")?;
    let lon_grid = array2_from_rows(lon_grid, "lon_grid")?;
    itu_rs_core::is_regular_grid(&lat_grid, &lon_grid).map_err(into_py_err)
}

#[pyfunction]
fn nearest_2d_interpolate(
    lat_grid: Vec<Vec<f64>>,
    lon_grid: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    lat_deg: f64,
    lon_deg: f64,
) -> PyResult<f64> {
    let lat_grid = array2_from_rows(lat_grid, "lat_grid")?;
    let lon_grid = array2_from_rows(lon_grid, "lon_grid")?;
    let values = array2_from_rows(values, "values")?;
    itu_rs_core::nearest_2d_interpolate(&lat_grid, &lon_grid, &values, lat_deg, lon_deg)
        .map_err(into_py_err)
}

#[pyfunction]
fn bilinear_2d_interpolate(
    lat_grid: Vec<Vec<f64>>,
    lon_grid: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    lat_deg: f64,
    lon_deg: f64,
) -> PyResult<f64> {
    let lat_grid = array2_from_rows(lat_grid, "lat_grid")?;
    let lon_grid = array2_from_rows(lon_grid, "lon_grid")?;
    let values = array2_from_rows(values, "values")?;
    itu_rs_core::bilinear_2d_interpolate(&lat_grid, &lon_grid, &values, lat_deg, lon_deg)
        .map_err(into_py_err)
}

#[pyfunction]
fn bicubic_2d_interpolate(
    lat_grid: Vec<Vec<f64>>,
    lon_grid: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    lat_deg: f64,
    lon_deg: f64,
) -> PyResult<f64> {
    let lat_grid = array2_from_rows(lat_grid, "lat_grid")?;
    let lon_grid = array2_from_rows(lon_grid, "lon_grid")?;
    let values = array2_from_rows(values, "values")?;
    itu_rs_core::bicubic_2d_interpolate(&lat_grid, &lon_grid, &values, lat_deg, lon_deg)
        .map_err(into_py_err)
}

py_result_fn!(fn topographic_altitude_km(lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::topographic_altitude_km);
py_result_fn!(fn surface_mean_temperature_k(lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::surface_mean_temperature_k);
py_result_fn!(fn surface_month_mean_temperature_k(lat_deg: f64, lon_deg: f64, month: u8) -> f64 => itu_rs_core::surface_month_mean_temperature_k);
py_result_fn!(fn standard_temperature_k(h_km: f64) -> f64 => itu_rs_core::standard_temperature_k);
py_result_fn!(fn standard_pressure_hpa(h_km: f64) -> f64 => itu_rs_core::standard_pressure_hpa);
py_result_fn!(fn standard_water_vapour_density_gm3(h_km: f64, rho0_gm3: f64) -> f64 => itu_rs_core::standard_water_vapour_density_gm3);
py_result_fn!(fn surface_water_vapour_density_gm3(lat_deg: f64, lon_deg: f64, p: f64, alt_km: f64) -> f64 => itu_rs_core::surface_water_vapour_density_gm3);
py_result_fn!(fn total_water_vapour_content_kgm2(lat_deg: f64, lon_deg: f64, p: f64, alt_km: f64) -> f64 => itu_rs_core::total_water_vapour_content_kgm2);
py_result_fn!(fn rainfall_rate_r001_mmh(lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::rainfall_rate_r001_mmh);
py_result_fn!(fn rainfall_probability_percent(lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::rainfall_probability_percent);
py_result_fn!(fn rainfall_rate_mmh(lat_deg: f64, lon_deg: f64, p: f64) -> f64 => itu_rs_core::rainfall_rate_mmh);
py_result_fn!(fn unavailability_from_rainfall_rate_percent(lat_deg: f64, lon_deg: f64, rainfall_rate_mmh: f64) -> f64 => itu_rs_core::unavailability_from_rainfall_rate_percent);
py_result_fn!(fn zero_isotherm_height_km(lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::zero_isotherm_height_km);
py_result_fn!(fn rain_height_km(lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::rain_height_km);
py_result_fn!(fn rain_specific_attenuation_coefficients(freq_ghz: f64, elevation_deg: f64, tau_deg: f64) -> (f64, f64) => itu_rs_core::rain_specific_attenuation_coefficients);
py_result_fn!(fn rain_specific_attenuation_db_per_km(rainfall_rate_mmh: f64, freq_ghz: f64, elevation_deg: f64, tau_deg: f64) -> f64 => itu_rs_core::rain_specific_attenuation_db_per_km);
py_result_fn!(fn cloud_reduced_liquid_kgm2(lat_deg: f64, lon_deg: f64, p: f64) -> f64 => itu_rs_core::cloud_reduced_liquid_kgm2);
py_result_fn!(fn cloud_liquid_mass_absorption_coefficient(freq_ghz: f64) -> f64 => itu_rs_core::cloud_liquid_mass_absorption_coefficient);
py_result_fn!(fn cloud_specific_attenuation_coefficient(freq_ghz: f64, temp_c: f64) -> f64 => itu_rs_core::cloud_specific_attenuation_coefficient);
py_result_fn!(
    #[pyo3(signature = (lat_deg, lon_deg, elevation_deg, freq_ghz, p, lred_kgm2=None))]
    fn cloud_attenuation_db(lat_deg: f64, lon_deg: f64, elevation_deg: f64, freq_ghz: f64, p: f64, lred_kgm2: Option<f64>) -> f64 => itu_rs_core::cloud_attenuation_db
);
py_result_fn!(fn lognormal_approximation_coefficients(lat_deg: f64, lon_deg: f64) -> (f64, f64, f64) => itu_rs_core::lognormal_approximation_coefficients);
py_result_fn!(fn cloud_attenuation_lognormal_db(lat_deg: f64, lon_deg: f64, elevation_deg: f64, freq_ghz: f64, p: f64) -> f64 => itu_rs_core::cloud_attenuation_lognormal_db);
py_result_fn!(fn wet_term_radio_refractivity(e_hpa: f64, temp_c: f64) -> f64 => itu_rs_core::wet_term_radio_refractivity);
py_result_fn!(fn dry_term_radio_refractivity(pd_hpa: f64, temp_k: f64) -> f64 => itu_rs_core::dry_term_radio_refractivity);
py_result_fn!(fn radio_refractive_index(pd_hpa: f64, e_hpa: f64, temp_k: f64) -> f64 => itu_rs_core::radio_refractive_index);
py_result_fn!(fn water_vapour_pressure_hpa(temp_c: f64, pressure_hpa: f64, humidity_percent: f64) -> f64 => itu_rs_core::water_vapour_pressure_hpa);
py_result_fn!(fn map_wet_term_radio_refractivity(lat_deg: f64, lon_deg: f64, p: f64) -> f64 => itu_rs_core::map_wet_term_radio_refractivity);
py_result_fn!(fn dn65(lat_deg: f64, lon_deg: f64, p: f64) -> f64 => itu_rs_core::dn65);
py_result_fn!(fn dn1(lat_deg: f64, lon_deg: f64, p: f64) -> f64 => itu_rs_core::dn1);
py_result_fn!(fn inter_annual_variability(p_fraction: f64, lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::inter_annual_variability);
py_result_fn!(fn risk_of_exceedance(p_fraction: f64, pr_fraction: f64, lat_deg: f64, lon_deg: f64) -> f64 => itu_rs_core::risk_of_exceedance);
py_result_fn!(fn gamma0_exact_db_per_km(freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 => itu_rs_core::gamma0_exact_db_per_km);
py_result_fn!(fn gammaw_exact_db_per_km(freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 => itu_rs_core::gammaw_exact_db_per_km);
py_result_fn!(fn gamma_exact_db_per_km(freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 => itu_rs_core::gamma_exact_db_per_km);
py_result_fn!(fn gamma0_approx_db_per_km(freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 => itu_rs_core::gamma0_approx_db_per_km);
py_result_fn!(fn gammaw_approx_db_per_km(freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> f64 => itu_rs_core::gammaw_approx_db_per_km);
py_result_fn!(fn slant_inclined_path_equivalent_height_km(freq_ghz: f64, pressure_hpa: f64, rho_gm3: f64, temp_k: f64) -> (f64, f64) => itu_rs_core::slant_inclined_path_equivalent_height_km);
py_result_fn!(fn zenith_water_vapour_attenuation_db(freq_ghz: f64, v_t_kgm2: f64, h_km: f64) -> f64 => itu_rs_core::zenith_water_vapour_attenuation_db);
py_result_fn!(fn gaseous_attenuation_slant_path_db(freq_ghz: f64, elevation_deg: f64, rho_gm3: f64, pressure_hpa: f64, temp_k: f64, v_t_kgm2: f64, h_km: f64, exact: bool) -> f64 => itu_rs_core::gaseous_attenuation_slant_path_db);
py_result_fn!(fn rain_attenuation_probability_percent(lat_deg: f64, lon_deg: f64, elevation_deg: f64, hs_km: Option<f64>, l_s_km: Option<f64>, p0_percent: Option<f64>) -> f64 => itu_rs_core::rain_attenuation_probability_percent);
py_result_fn!(fn fit_rain_attenuation_to_lognormal(lat_deg: f64, lon_deg: f64, freq_ghz: f64, elevation_deg: f64, hs_km: f64, p_k_percent: f64, tau_deg: f64) -> (f64, f64) => itu_rs_core::fit_rain_attenuation_to_lognormal);
py_result_fn!(fn rain_attenuation_db(lat_deg: f64, lon_deg: f64, freq_ghz: f64, elevation_deg: f64, hs_km: f64, p: f64, r001_mmh: Option<f64>, tau_deg: f64, l_s_km: Option<f64>) -> f64 => itu_rs_core::rain_attenuation_db);
py_result_fn!(fn site_diversity_rain_outage_probability_percent(lat1_deg: f64, lon1_deg: f64, a1_db: f64, elevation1_deg: f64, lat2_deg: f64, lon2_deg: f64, a2_db: f64, elevation2_deg: f64, freq_ghz: f64, tau_deg: f64, hs1_km: Option<f64>, hs2_km: Option<f64>) -> f64 => itu_rs_core::site_diversity_rain_outage_probability_percent);
py_result_fn!(fn rain_cross_polarization_discrimination_db(attenuation_db: f64, freq_ghz: f64, elevation_deg: f64, p: f64, tau_deg: f64) -> f64 => itu_rs_core::rain_cross_polarization_discrimination_db);
py_result_fn!(fn scintillation_sigma_db(lat_deg: f64, lon_deg: f64, freq_ghz: f64, elevation_deg: f64, dish_m: f64, eta: f64, temp_c: Option<f64>, humidity_percent: Option<f64>, pressure_hpa: Option<f64>, h_l_m: f64) -> f64 => itu_rs_core::scintillation_sigma_db);
py_result_fn!(fn scintillation_attenuation_db(lat_deg: f64, lon_deg: f64, freq_ghz: f64, elevation_deg: f64, p: f64, dish_m: f64, eta: f64, temp_c: Option<f64>, humidity_percent: Option<f64>, pressure_hpa: Option<f64>, h_l_m: f64) -> f64 => itu_rs_core::scintillation_attenuation_db);
py_result_fn!(fn gas_attenuation_default(lat_deg: f64, lon_deg: f64, freq_ghz: f64, elevation_deg: f64, p: f64, d_m: f64) -> f64 => itu_rs_core::gas_attenuation_default);

#[pyfunction]
fn saturation_vapour_pressure_hpa(
    temp_c: f64,
    pressure_hpa: f64,
    hydrometeor: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    itu_rs_core::saturation_vapour_pressure_hpa(
        temp_c,
        pressure_hpa,
        parse_hydrometeor(hydrometeor)?,
    )
    .map_err(into_py_err)
}

#[pyfunction]
fn gaseous_attenuation_terrestrial_path_db(
    path_length_km: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    rho_gm3: f64,
    pressure_hpa: f64,
    temp_k: f64,
    mode: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    itu_rs_core::gaseous_attenuation_terrestrial_path_db(
        path_length_km,
        freq_ghz,
        elevation_deg,
        rho_gm3,
        pressure_hpa,
        temp_k,
        parse_gas_path_mode(mode)?,
    )
    .map_err(into_py_err)
}

#[pyfunction]
fn gaseous_attenuation_inclined_path_db(
    freq_ghz: f64,
    elevation_deg: f64,
    rho_gm3: f64,
    pressure_hpa: f64,
    temp_k: f64,
    h1_km: f64,
    h2_km: f64,
    mode: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    itu_rs_core::gaseous_attenuation_inclined_path_db(
        freq_ghz,
        elevation_deg,
        rho_gm3,
        pressure_hpa,
        temp_k,
        h1_km,
        h2_km,
        parse_gas_path_mode(mode)?,
    )
    .map_err(into_py_err)
}

#[pyfunction]
fn gas_attenuation_default_many(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: Vec<f64>,
    p: f64,
    d_m: f64,
) -> PyResult<Vec<f64>> {
    itu_rs_core::gas_attenuation_default_many(lat_deg, lon_deg, freq_ghz, &elevation_deg, p, d_m)
        .map_err(into_py_err)
}

#[pyfunction]
fn gas_attenuation_default_many_checked(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: Vec<f64>,
    p: f64,
    d_m: f64,
) -> PyResult<Vec<f64>> {
    itu_rs_core::gas_attenuation_default_many_checked(
        lat_deg,
        lon_deg,
        freq_ghz,
        &elevation_deg,
        p,
        d_m,
    )
    .map_err(into_py_err)
}

#[pyfunction]
#[pyo3(signature = (lat_deg, lon_deg, freq_ghz, elevation_deg, p, d_m, options=None))]
fn atmospheric_attenuation_slant_path(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: f64,
    p: f64,
    d_m: f64,
    options: Option<PyRef<'_, PySlantPathOptions>>,
) -> PyResult<PySlantPathContributions> {
    let options = options
        .as_ref()
        .map(|options| options.to_core())
        .unwrap_or_default();
    itu_rs_core::atmospheric_attenuation_slant_path(
        lat_deg,
        lon_deg,
        freq_ghz,
        elevation_deg,
        p,
        d_m,
        options,
    )
    .map(PySlantPathContributions::from)
    .map_err(into_py_err)
}

#[pyfunction]
#[pyo3(signature = (lat_deg, lon_deg, freq_ghz, elevation_deg, p, d_m, options=None))]
fn atmospheric_attenuation_slant_path_many(
    lat_deg: f64,
    lon_deg: f64,
    freq_ghz: f64,
    elevation_deg: Vec<f64>,
    p: f64,
    d_m: f64,
    options: Option<PyRef<'_, PySlantPathOptions>>,
) -> PyResult<Vec<PySlantPathContributions>> {
    let options = options
        .as_ref()
        .map(|options| options.to_core())
        .unwrap_or_default();
    itu_rs_core::atmospheric_attenuation_slant_path_many(
        lat_deg,
        lon_deg,
        freq_ghz,
        &elevation_deg,
        p,
        d_m,
        options,
    )
    .map(|items| {
        items
            .into_iter()
            .map(PySlantPathContributions::from)
            .collect()
    })
    .map_err(into_py_err)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ItuError", m.py().get_type::<PyItuError>())?;
    m.add_class::<PyHydrometeorType>()?;
    m.add_class::<PySlantPathOptions>()?;
    m.add_class::<PySlantPathContributions>()?;

    add_py_functions!(
        m,
        [
            is_regular_grid,
            nearest_2d_interpolate,
            bilinear_2d_interpolate,
            bicubic_2d_interpolate,
            topographic_altitude_km,
            surface_mean_temperature_k,
            surface_month_mean_temperature_k,
            standard_temperature_k,
            standard_pressure_hpa,
            standard_water_vapour_density_gm3,
            surface_water_vapour_density_gm3,
            total_water_vapour_content_kgm2,
            rainfall_rate_r001_mmh,
            rainfall_probability_percent,
            rainfall_rate_mmh,
            unavailability_from_rainfall_rate_percent,
            zero_isotherm_height_km,
            rain_height_km,
            rain_specific_attenuation_coefficients,
            rain_specific_attenuation_db_per_km,
            cloud_reduced_liquid_kgm2,
            cloud_liquid_mass_absorption_coefficient,
            cloud_specific_attenuation_coefficient,
            cloud_attenuation_db,
            lognormal_approximation_coefficients,
            cloud_attenuation_lognormal_db,
            wet_term_radio_refractivity,
            dry_term_radio_refractivity,
            radio_refractive_index,
            water_vapour_pressure_hpa,
            saturation_vapour_pressure_hpa,
            map_wet_term_radio_refractivity,
            dn65,
            dn1,
            inter_annual_variability,
            risk_of_exceedance,
            gamma0_exact_db_per_km,
            gammaw_exact_db_per_km,
            gamma_exact_db_per_km,
            gamma0_approx_db_per_km,
            gammaw_approx_db_per_km,
            slant_inclined_path_equivalent_height_km,
            zenith_water_vapour_attenuation_db,
            gaseous_attenuation_slant_path_db,
            gaseous_attenuation_terrestrial_path_db,
            gaseous_attenuation_inclined_path_db,
            rain_attenuation_probability_percent,
            fit_rain_attenuation_to_lognormal,
            rain_attenuation_db,
            site_diversity_rain_outage_probability_percent,
            rain_cross_polarization_discrimination_db,
            scintillation_sigma_db,
            scintillation_attenuation_db,
            gas_attenuation_default,
            gas_attenuation_default_many,
            gas_attenuation_default_many_checked,
            atmospheric_attenuation_slant_path,
            atmospheric_attenuation_slant_path_many,
        ]
    );

    Ok(())
}
