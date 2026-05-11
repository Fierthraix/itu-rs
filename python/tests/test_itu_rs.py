import math

import pytest

import itu_rs


def test_scalar_api_uses_embedded_data():
    value = itu_rs.topographic_altitude_km(45.4215, -75.6972)

    assert math.isfinite(value)


def test_slant_path_api_returns_contributions():
    attenuation = itu_rs.atmospheric_attenuation_slant_path(
        45.4215,
        -75.6972,
        12.0,
        30.0,
        0.1,
        1.2,
        itu_rs.SlantPathOptions.default(),
    )

    assert math.isfinite(attenuation.total_db)
    assert math.isfinite(attenuation.gas_db)


def test_batch_api_accepts_python_sequences():
    gas = itu_rs.gas_attenuation_default_many(
        45.4215,
        -75.6972,
        12.0,
        [5.0, 30.0, 89.0],
        0.1,
        1.2,
    )

    assert len(gas) == 3
    assert all(math.isfinite(value) for value in gas)


def test_hydrometeor_accepts_class_or_string():
    water = itu_rs.saturation_vapour_pressure_hpa(
        15.0,
        1000.0,
        itu_rs.HydrometeorType.water(),
    )
    ice = itu_rs.saturation_vapour_pressure_hpa(15.0, 1000.0, "ice")

    assert water > 0.0
    assert ice > 0.0


def test_optional_arguments_have_python_defaults():
    cloud = itu_rs.cloud_attenuation_db(45.4215, -75.6972, 30.0, 12.0, 1.0)

    assert cloud >= 0.0


def test_new_recommendation_helpers_are_exposed():
    lat = 45.4215
    lon = -75.6972

    terrestrial = itu_rs.gaseous_attenuation_terrestrial_path_db(
        10.0,
        12.0,
        30.0,
        7.5,
        1008.0,
        289.2,
        "approximate",
    )
    inclined = itu_rs.gaseous_attenuation_inclined_path_db(
        12.0,
        30.0,
        7.5,
        1008.0,
        289.2,
        0.4,
        2.0,
        "exact",
    )
    rain_probability = itu_rs.rain_attenuation_probability_percent(
        lat,
        lon,
        30.0,
        0.4,
        None,
        None,
    )
    sigma, mean = itu_rs.fit_rain_attenuation_to_lognormal(
        lat,
        lon,
        12.0,
        30.0,
        0.4,
        10.0,
        45.0,
    )
    xpd = itu_rs.rain_cross_polarization_discrimination_db(10.0, 12.0, 30.0, 0.1, 45.0)

    assert terrestrial >= 0.0
    assert inclined >= 0.0
    assert rain_probability >= 0.0
    assert math.isfinite(sigma)
    assert math.isfinite(mean)
    assert math.isfinite(xpd)


def test_regular_grid_helpers_accept_python_sequences():
    lat = [[1.0, 1.0], [0.0, 0.0]]
    lon = [[0.0, 1.0], [0.0, 1.0]]
    values = [[10.0, 11.0], [0.0, 1.0]]

    assert itu_rs.is_regular_grid(lat, lon)
    assert itu_rs.nearest_2d_interpolate(lat, lon, values, 0.2, 0.2) == 0.0
    assert math.isclose(
        itu_rs.bilinear_2d_interpolate(lat, lon, values, 0.5, 0.5),
        5.5,
    )


def test_invalid_input_raises_itu_error():
    with pytest.raises(itu_rs.ItuError):
        itu_rs.topographic_altitude_km(100.0, 0.0)
