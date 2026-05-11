use criterion::{Criterion, criterion_group, criterion_main};
use itu_rs::{
    GasPathMode, HydrometeorType, SlantPathOptions, atmospheric_attenuation_slant_path,
    atmospheric_attenuation_slant_path_many, cloud_attenuation_lognormal_db, dn1, dn65,
    dry_term_radio_refractivity, fit_rain_attenuation_to_lognormal,
    gaseous_attenuation_inclined_path_db, gaseous_attenuation_terrestrial_path_db,
    inter_annual_variability, lognormal_approximation_coefficients,
    rain_attenuation_probability_percent, rain_cross_polarization_discrimination_db,
    rainfall_probability_percent, rainfall_rate_mmh, risk_of_exceedance,
    saturation_vapour_pressure_hpa, surface_month_mean_temperature_k,
    unavailability_from_rainfall_rate_percent, zero_isotherm_height_km,
};
use std::hint::black_box;

fn bench_default_many(c: &mut Criterion) {
    let elevations: Vec<f64> = (0..169)
        .map(|idx| 5.0 + idx as f64 * (89.0 - 5.0) / 168.0)
        .collect();

    c.bench_function("default_many_169_elevations", |b| {
        b.iter(|| {
            atmospheric_attenuation_slant_path_many(
                black_box(45.4215),
                black_box(-75.6972),
                black_box(12.0),
                black_box(&elevations),
                black_box(0.1),
                black_box(1.2),
                black_box(SlantPathOptions::default()),
            )
            .unwrap()
        })
    });
}

fn bench_exact_scalar(c: &mut Criterion) {
    let options = SlantPathOptions {
        hs_km: Some(0.4),
        rho_gm3: Some(7.5),
        r001_mmh: Some(28.0),
        eta: 0.67,
        t: Some(289.2),
        h_percent: Some(61.0),
        pressure_hpa: Some(1008.0),
        h_l_m: 900.0,
        l_s_km: Some(6.7),
        tau_deg: 33.0,
        v_t_kgm2: Some(22.5),
        exact: true,
        ..SlantPathOptions::default()
    };

    c.bench_function("exact_scalar_with_contributions", |b| {
        b.iter(|| {
            atmospheric_attenuation_slant_path(
                black_box(10.0),
                black_box(20.0),
                black_box(18.0),
                black_box(17.5),
                black_box(0.7),
                black_box(0.8),
                black_box(options),
            )
            .unwrap()
        })
    });
}

fn bench_scalar_recommendation_apis(c: &mut Criterion) {
    c.bench_function("scalar_recommendation_api_set", |b| {
        b.iter(|| {
            let lat = black_box(45.4215);
            let lon = black_box(-75.6972);
            (
                surface_month_mean_temperature_k(lat, lon, 1).unwrap(),
                rainfall_probability_percent(lat, lon).unwrap(),
                rainfall_rate_mmh(lat, lon, 0.1).unwrap(),
                unavailability_from_rainfall_rate_percent(lat, lon, 10.0).unwrap(),
                zero_isotherm_height_km(lat, lon).unwrap(),
                dry_term_radio_refractivity(1000.0, 288.15).unwrap(),
                saturation_vapour_pressure_hpa(15.0, 1000.0, HydrometeorType::Water).unwrap(),
                dn65(lat, lon, 50.0).unwrap(),
                dn1(lat, lon, 50.0).unwrap(),
                inter_annual_variability(0.001, lat, lon).unwrap(),
                risk_of_exceedance(0.001, 0.001, lat, lon).unwrap(),
                lognormal_approximation_coefficients(lat, lon).unwrap(),
                cloud_attenuation_lognormal_db(lat, lon, 30.0, 12.0, 1.0).unwrap(),
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
                rain_attenuation_probability_percent(lat, lon, 30.0, Some(0.4), None, None)
                    .unwrap(),
                fit_rain_attenuation_to_lognormal(lat, lon, 12.0, 30.0, 0.4, 10.0, 45.0).unwrap(),
                rain_cross_polarization_discrimination_db(10.0, 12.0, 30.0, 0.1, 45.0).unwrap(),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_default_many,
    bench_exact_scalar,
    bench_scalar_recommendation_apis
);
criterion_main!(benches);
