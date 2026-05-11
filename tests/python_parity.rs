use itu_rs::{
    GasPathMode, HydrometeorType, cloud_attenuation_lognormal_db, dn1, dn65,
    dry_term_radio_refractivity, fit_rain_attenuation_to_lognormal, gamma0_approx_db_per_km,
    gammaw_approx_db_per_km, gaseous_attenuation_inclined_path_db,
    gaseous_attenuation_terrestrial_path_db, inter_annual_variability,
    lognormal_approximation_coefficients, rain_attenuation_probability_percent,
    rain_cross_polarization_discrimination_db, rainfall_probability_percent, rainfall_rate_mmh,
    risk_of_exceedance, saturation_vapour_pressure_hpa,
    site_diversity_rain_outage_probability_percent, surface_month_mean_temperature_k,
    unavailability_from_rainfall_rate_percent, zero_isotherm_height_km,
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

    assert_close(
        gamma0_approx_db_per_km(12.0, 1008.0, 7.5, 289.2).unwrap(),
        0.017926109900869132,
        1e-12,
    );
    assert_close(
        gammaw_approx_db_per_km(12.0, 1008.0, 7.5, 289.2).unwrap(),
        0.017926109900869132,
        1e-12,
    );
    assert_close(
        gaseous_attenuation_terrestrial_path_db(
            10.0,
            12.0,
            30.0,
            7.5,
            1008.0,
            289.2,
            GasPathMode::Approximate,
        )
        .unwrap(),
        0.1792610990086913,
        1e-12,
    );
    assert_close(
        gaseous_attenuation_inclined_path_db(
            12.0,
            30.0,
            7.5,
            1008.0,
            289.2,
            0.4,
            2.0,
            GasPathMode::Approximate,
        )
        .unwrap(),
        0.042065152883089964,
        1e-5,
    );
    assert_close(
        gaseous_attenuation_inclined_path_db(
            12.0,
            30.0,
            7.5,
            1008.0,
            289.2,
            0.4,
            2.0,
            GasPathMode::Exact,
        )
        .unwrap(),
        0.021566039863650292,
        1e-5,
    );
    assert_close(
        rain_attenuation_probability_percent(lat, lon, 30.0, Some(0.4), None, None).unwrap(),
        10.833134985735281,
        1e-5,
    );
    assert_close(
        rain_cross_polarization_discrimination_db(10.0, 12.0, 30.0, 0.1, 45.0).unwrap(),
        12.83971164251977,
        1e-12,
    );
    let (rain_sigma, rain_mean) =
        fit_rain_attenuation_to_lognormal(lat, lon, 12.0, 30.0, 0.4, 10.0, 45.0).unwrap();
    assert_close(rain_sigma, 1.30698123, 1e-8);
    assert_close(rain_mean, -2.32980755, 1e-8);
    assert_close(
        site_diversity_rain_outage_probability_percent(
            lat,
            lon,
            5.0,
            30.0,
            lat + 0.2,
            lon + 0.3,
            5.0,
            30.0,
            12.0,
            45.0,
            Some(0.4),
            Some(0.4),
        )
        .unwrap(),
        0.0001658916749279343,
        5e-5,
    );
}
