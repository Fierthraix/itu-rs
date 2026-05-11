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


def test_invalid_input_raises_itu_error():
    with pytest.raises(itu_rs.ItuError):
        itu_rs.topographic_altitude_km(100.0, 0.0)
