use itu_rs::{
    SlantPathOptions, atmospheric_attenuation_slant_path, atmospheric_attenuation_slant_path_many,
    gas_attenuation_default, gas_attenuation_default_many_checked,
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
