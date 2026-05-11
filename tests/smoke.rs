use itu_rs::{
    HydrometeorType, SlantPathOptions, atmospheric_attenuation_slant_path,
    atmospheric_attenuation_slant_path_many, cloud_attenuation_db, cloud_attenuation_lognormal_db,
    cloud_liquid_mass_absorption_coefficient, cloud_reduced_liquid_kgm2,
    cloud_specific_attenuation_coefficient, dn1, dn65, dry_term_radio_refractivity,
    gamma_exact_db_per_km, gamma0_exact_db_per_km, gammaw_exact_db_per_km, gas_attenuation_default,
    gas_attenuation_default_many_checked, gaseous_attenuation_slant_path_db,
    inter_annual_variability, lognormal_approximation_coefficients,
    map_wet_term_radio_refractivity, radio_refractive_index, rain_attenuation_db, rain_height_km,
    rain_specific_attenuation_coefficients, rain_specific_attenuation_db_per_km,
    rainfall_probability_percent, rainfall_rate_mmh, rainfall_rate_r001_mmh, risk_of_exceedance,
    saturation_vapour_pressure_hpa, scintillation_attenuation_db, scintillation_sigma_db,
    slant_inclined_path_equivalent_height_km, standard_pressure_hpa, standard_temperature_k,
    standard_water_vapour_density_gm3, surface_mean_temperature_k,
    surface_month_mean_temperature_k, surface_water_vapour_density_gm3, topographic_altitude_km,
    total_water_vapour_content_kgm2, water_vapour_pressure_hpa, wet_term_radio_refractivity,
    zenith_water_vapour_attenuation_db, zero_isotherm_height_km,
};
use std::path::Path;

fn data_available() -> bool {
    cfg!(feature = "data")
        || std::env::var_os("ITU_RS_DATA_DIR").is_some()
        || Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data/1511/v2_lat.npz")
            .exists()
}

#[test]
fn gas_scalar_and_many_are_consistent() {
    if !data_available() {
        return;
    }

    let elevations = [5.0, 30.0, 89.0];
    let many = gas_attenuation_default_many_checked(45.4215, -75.6972, 12.0, &elevations, 0.1, 1.2)
        .unwrap();

    for (idx, elevation) in elevations.iter().enumerate() {
        let scalar =
            gas_attenuation_default(45.4215, -75.6972, 12.0, *elevation, 0.1, 1.2).unwrap();
        assert!((many[idx] - scalar).abs() < 1e-12);
    }
}

#[test]
fn slant_path_returns_contributions() {
    if !data_available() {
        return;
    }

    let result = atmospheric_attenuation_slant_path(
        45.4215,
        -75.6972,
        12.0,
        30.0,
        0.1,
        1.2,
        SlantPathOptions::default(),
    )
    .unwrap();

    assert!(result.total_db.is_finite());
    assert!(result.gas_db.is_finite());
    assert!(result.cloud_db.is_finite());
    assert!(result.rain_db.is_finite());
    assert!(result.scintillation_db.is_finite());
}

#[test]
fn slant_path_many_matches_scalar() {
    if !data_available() {
        return;
    }

    let elevations = [5.0, 17.5, 45.0, 89.0];
    let options = SlantPathOptions::default();
    let many = atmospheric_attenuation_slant_path_many(
        45.4215,
        -75.6972,
        12.0,
        &elevations,
        0.1,
        1.2,
        options,
    )
    .unwrap();

    for (idx, elevation) in elevations.iter().enumerate() {
        let scalar = atmospheric_attenuation_slant_path(
            45.4215, -75.6972, 12.0, *elevation, 0.1, 1.2, options,
        )
        .unwrap();
        assert!((many[idx].total_db - scalar.total_db).abs() < 1e-12);
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let err = atmospheric_attenuation_slant_path(
        91.0,
        -75.6972,
        12.0,
        30.0,
        0.1,
        1.2,
        SlantPathOptions::default(),
    )
    .unwrap_err();

    assert_eq!(err.message(), "lat_deg must be in [-90, 90]");
}

#[test]
fn direct_lookup_wrappers_return_finite_values() {
    if !data_available() {
        return;
    }

    let lat = 45.4215;
    let lon = -75.6972;
    let p = 1.0;
    let h_km = topographic_altitude_km(lat, lon).unwrap();

    let values = [
        h_km,
        surface_mean_temperature_k(lat, lon).unwrap(),
        surface_month_mean_temperature_k(lat, lon, 1).unwrap(),
        standard_temperature_k(h_km).unwrap(),
        standard_pressure_hpa(h_km).unwrap(),
        standard_water_vapour_density_gm3(h_km, 7.5).unwrap(),
        surface_water_vapour_density_gm3(lat, lon, p, h_km).unwrap(),
        total_water_vapour_content_kgm2(lat, lon, p, h_km).unwrap(),
        rainfall_rate_r001_mmh(lat, lon).unwrap(),
        rainfall_probability_percent(lat, lon).unwrap(),
        rainfall_rate_mmh(lat, lon, 0.1).unwrap(),
        unavailability_from_rainfall_rate_percent_for_test(lat, lon, 10.0),
        zero_isotherm_height_km(lat, lon).unwrap(),
        rain_height_km(lat, lon).unwrap(),
        cloud_reduced_liquid_kgm2(lat, lon, p).unwrap(),
        cloud_liquid_mass_absorption_coefficient(12.0).unwrap(),
        cloud_specific_attenuation_coefficient(12.0, 0.6).unwrap(),
        cloud_attenuation_lognormal_db(lat, lon, 30.0, 12.0, p).unwrap(),
        wet_term_radio_refractivity(12.0, 15.0).unwrap(),
        dry_term_radio_refractivity(1000.0, 288.15).unwrap(),
        radio_refractive_index(1000.0, 12.0, 288.15).unwrap(),
        water_vapour_pressure_hpa(15.0, 1000.0, 60.0).unwrap(),
        saturation_vapour_pressure_hpa(15.0, 1000.0, HydrometeorType::Water).unwrap(),
        map_wet_term_radio_refractivity(lat, lon, 50.0).unwrap(),
        dn65(lat, lon, 50.0).unwrap(),
        dn1(lat, lon, 50.0).unwrap(),
        inter_annual_variability(0.001, lat, lon).unwrap(),
        risk_of_exceedance(0.001, 0.001, lat, lon).unwrap(),
        gamma0_exact_db_per_km(12.0, 1008.0, 7.5, 289.2).unwrap(),
        gammaw_exact_db_per_km(12.0, 1008.0, 7.5, 289.2).unwrap(),
        gamma_exact_db_per_km(12.0, 1008.0, 7.5, 289.2).unwrap(),
        zenith_water_vapour_attenuation_db(12.0, 22.5, h_km).unwrap(),
    ];

    assert!(values.iter().all(|value| value.is_finite()));

    let (h0, hw) = slant_inclined_path_equivalent_height_km(12.0, 1008.0, 7.5, 289.2).unwrap();
    assert!(h0.is_finite());
    assert!(hw.is_finite());

    let (k, alpha) = rain_specific_attenuation_coefficients(12.0, 30.0, 45.0).unwrap();
    assert!(k.is_finite());
    assert!(alpha.is_finite());

    let (_m, sigma, pclw) = lognormal_approximation_coefficients(lat, lon).unwrap();
    assert!(sigma.is_finite());
    assert!(pclw.is_finite());
}

fn unavailability_from_rainfall_rate_percent_for_test(lat: f64, lon: f64, rain: f64) -> f64 {
    itu_rs::unavailability_from_rainfall_rate_percent(lat, lon, rain).unwrap()
}

#[test]
fn rain_specific_attenuation_matches_coefficients() {
    let rainfall_rate = 28.0;
    let (k, alpha) = rain_specific_attenuation_coefficients(12.0, 30.0, 45.0).unwrap();
    let gamma = rain_specific_attenuation_db_per_km(rainfall_rate, 12.0, 30.0, 45.0).unwrap();
    assert!((gamma - k * rainfall_rate.powf(alpha)).abs() < 1e-12);
}

#[test]
fn direct_component_wrappers_match_slant_path_contributions() {
    if !data_available() {
        return;
    }

    let lat = 45.4215;
    let lon = -75.6972;
    let freq = 12.0;
    let elevation = 30.0;
    let p = 1.0;
    let dish_m = 1.2;

    let gas_options = SlantPathOptions {
        hs_km: Some(0.4),
        rho_gm3: Some(7.5),
        t: Some(289.2),
        pressure_hpa: Some(1008.0),
        v_t_kgm2: Some(22.5),
        include_rain: false,
        include_clouds: false,
        include_scintillation: false,
        ..SlantPathOptions::default()
    };
    let gas = atmospheric_attenuation_slant_path(lat, lon, freq, elevation, p, dish_m, gas_options)
        .unwrap();
    let direct_gas =
        gaseous_attenuation_slant_path_db(freq, elevation, 7.5, 1008.0, 289.2, 22.5, 0.4, false)
            .unwrap();
    assert!((gas.gas_db - direct_gas).abs() < 1e-12);

    let rain_options = SlantPathOptions {
        hs_km: Some(0.4),
        r001_mmh: Some(28.0),
        tau_deg: 33.0,
        l_s_km: Some(6.7),
        include_gas: false,
        include_clouds: false,
        include_scintillation: false,
        ..SlantPathOptions::default()
    };
    let rain =
        atmospheric_attenuation_slant_path(lat, lon, freq, elevation, p, dish_m, rain_options)
            .unwrap();
    let direct_rain = rain_attenuation_db(
        lat,
        lon,
        freq,
        elevation,
        0.4,
        p,
        Some(28.0),
        33.0,
        Some(6.7),
    )
    .unwrap();
    assert!((rain.rain_db - direct_rain).abs() < 1e-12);

    let cloud_options = SlantPathOptions {
        include_rain: false,
        include_gas: false,
        include_scintillation: false,
        ..SlantPathOptions::default()
    };
    let cloud =
        atmospheric_attenuation_slant_path(lat, lon, freq, elevation, p, dish_m, cloud_options)
            .unwrap();
    let direct_cloud = cloud_attenuation_db(lat, lon, elevation, freq, p, None).unwrap();
    assert!((cloud.cloud_db - direct_cloud).abs() < 1e-12);

    let scintillation_options = SlantPathOptions {
        include_rain: false,
        include_gas: false,
        include_clouds: false,
        ..SlantPathOptions::default()
    };
    let scintillation = atmospheric_attenuation_slant_path(
        lat,
        lon,
        freq,
        elevation,
        p,
        dish_m,
        scintillation_options,
    )
    .unwrap();
    let sigma = scintillation_sigma_db(
        lat, lon, freq, elevation, dish_m, 0.5, None, None, None, 1000.0,
    )
    .unwrap();
    let direct_scintillation = scintillation_attenuation_db(
        lat, lon, freq, elevation, p, dish_m, 0.5, None, None, None, 1000.0,
    )
    .unwrap();
    assert!(sigma.is_finite());
    assert!((scintillation.scintillation_db - direct_scintillation).abs() < 1e-12);
}

#[test]
fn direct_wrapper_invalid_inputs_are_rejected() {
    let err = rainfall_rate_r001_mmh(91.0, 0.0).unwrap_err();
    assert_eq!(err.message(), "lat_deg must be in [-90, 90]");

    let err = cloud_attenuation_db(0.0, 0.0, 0.0, 12.0, 1.0, None).unwrap_err();
    assert_eq!(err.message(), "elevation_deg must be in (0, 90)");

    let err = surface_month_mean_temperature_k(0.0, 0.0, 13).unwrap_err();
    assert_eq!(err.message(), "month must be in 1..=12");

    let err = risk_of_exceedance(0.05, 0.05, 0.0, 0.0).unwrap_err();
    assert_eq!(err.message(), "p_fraction must be in [0.0001, 0.02]");

    let err = scintillation_sigma_db(
        0.0,
        0.0,
        12.0,
        30.0,
        1.2,
        0.5,
        Some(10.0),
        None,
        Some(1000.0),
        1000.0,
    )
    .unwrap_err();
    assert_eq!(
        err.message(),
        "temp_c, humidity_percent, and pressure_hpa must be supplied together"
    );
}
