use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itu_rs::{
    atmospheric_attenuation_slant_path, atmospheric_attenuation_slant_path_many, SlantPathOptions,
};

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

criterion_group!(benches, bench_default_many, bench_exact_scalar);
criterion_main!(benches);
