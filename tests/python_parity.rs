use itu_rs::{
    HydrometeorType, cloud_attenuation_lognormal_db, dn1, dn65, dry_term_radio_refractivity,
    inter_annual_variability, lognormal_approximation_coefficients, rainfall_probability_percent,
    rainfall_rate_mmh, risk_of_exceedance, saturation_vapour_pressure_hpa,
    surface_month_mean_temperature_k, unavailability_from_rainfall_rate_percent,
    zero_isotherm_height_km,
};
use std::path::Path;

fn data_available() -> bool {
    cfg!(feature = "data")
        || std::env::var_os("ITU_RS_DATA_DIR").is_some()
        || Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data/1511/v2_lat.npz")
            .exists()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:.17e} != expected {expected:.17e} within {tolerance:.3e}"
    );
}

#[test]
fn scalar_recommendation_reference_values_match_python_itu_r() {
    if !data_available() {
        return;
    }

    let lat = 45.4215;
    let lon = -75.6972;

    // Reference values generated from ~/pkg/ITU-Rpy at the same site.
    assert_close(
        surface_month_mean_temperature_k(lat, lon, 1).unwrap(),
        263.5091672896,
        1e-9,
    );
    assert_close(
        rainfall_probability_percent(lat, lon).unwrap(),
        7.738488927221891,
        1e-9,
    );
    assert_close(
        rainfall_rate_mmh(lat, lon, 0.1).unwrap(),
        10.311849415401246,
        1e-5,
    );
    assert_close(
        unavailability_from_rainfall_rate_percent(lat, lon, 10.0).unwrap(),
        0.1056168360918215,
        1e-5,
    );
    assert_close(
        zero_isotherm_height_km(lat, lon).unwrap(),
        3.2101220736,
        1e-10,
    );
    assert_close(
        dry_term_radio_refractivity(1000.0, 288.15).unwrap(),
        269.30418184973104,
        1e-12,
    );
    assert_close(
        saturation_vapour_pressure_hpa(15.0, 1000.0, HydrometeorType::Water).unwrap(),
        17.12083475316733,
        1e-12,
    );
    assert_close(
        saturation_vapour_pressure_hpa(15.0, 1000.0, HydrometeorType::Ice).unwrap(),
        19.768853236765214,
        1e-12,
    );
    assert_close(dn65(lat, lon, 50.0).unwrap(), -45.6743520208, 1e-9);
    assert_close(dn1(lat, lon, 50.0).unwrap(), -34.21593732096, 1e-9);
    assert_close(
        inter_annual_variability(0.001, lat, lon).unwrap(),
        7.929539514889553e-8,
        1e-15,
    );
    assert_close(
        risk_of_exceedance(0.001, 0.001, lat, lon).unwrap(),
        50.0,
        1e-9,
    );

    let (m, sigma, pclw) = lognormal_approximation_coefficients(lat, lon).unwrap();
    assert_close(m, -2.355741731199999, 1e-12);
    assert_close(sigma, 0.8992728992, 1e-12);
    assert_close(pclw, 51.09665370559998, 1e-9);
    assert_close(
        cloud_attenuation_lognormal_db(lat, lon, 30.0, 12.0, 1.0).unwrap(),
        0.14890147056652353,
        1e-9,
    );
}
