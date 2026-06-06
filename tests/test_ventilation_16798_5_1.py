"""Tests for the solver-independent EN 16798-5-1 sensible AHU model."""

import math

import pytest

from pybuildingenergy.source.ventilation_16798_5_1 import (
    CompensationPoint,
    SupplyTemperatureControl,
    SupplyTemperatureControlMode,
    resolve_supply_temperature_setpoint,
)


def _point(reference_c, supply_c):
    return CompensationPoint(reference_c, supply_c)


def test_fixed_supply_temperature_control():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.FIXED,
        setpoint_c=18.0,
    )

    assert resolve_supply_temperature_setpoint(control, -10.0, 22.0) == 18.0


def test_scheduled_supply_temperature_control():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.SCHEDULED,
    )

    assert (
        resolve_supply_temperature_setpoint(
            control,
            outdoor_temperature_c=-10.0,
            extract_temperature_c=22.0,
            scheduled_setpoint_c=17.5,
        )
        == 17.5
    )


def test_scheduled_control_requires_runtime_setpoint():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.SCHEDULED,
    )

    with pytest.raises(ValueError, match="requires scheduled_setpoint_c"):
        resolve_supply_temperature_setpoint(control, -10.0, 22.0)


def test_outdoor_compensation_interpolates_multiple_segments():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.OUTDOOR_COMPENSATED,
        points=(
            _point(10.0, 18.0),
            _point(-10.0, 22.0),
            _point(0.0, 20.0),
        ),
    )

    assert resolve_supply_temperature_setpoint(control, -5.0, 22.0) == 21.0
    assert resolve_supply_temperature_setpoint(control, 5.0, 22.0) == 19.0


@pytest.mark.parametrize(
    ("extract_temperature_c", "expected_supply_temperature_c"),
    [
        (20.0, 19.0),
        (21.0, 19.0),
        (22.0, 18.0),
        (23.0, 17.0),
        (24.0, 17.0),
    ],
)
def test_extract_compensation_matches_zeblab_curve(
    extract_temperature_c,
    expected_supply_temperature_c,
):
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
        points=(
            _point(21.0, 19.0),
            _point(23.0, 17.0),
        ),
    )

    assert (
        resolve_supply_temperature_setpoint(
            control,
            outdoor_temperature_c=-5.0,
            extract_temperature_c=extract_temperature_c,
        )
        == expected_supply_temperature_c
    )


def test_compensation_limits_apply_after_interpolation():
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.OUTDOOR_COMPENSATED,
        points=(
            _point(-10.0, 25.0),
            _point(10.0, 15.0),
        ),
        minimum_c=17.0,
        maximum_c=21.0,
    )

    assert resolve_supply_temperature_setpoint(control, -10.0, 22.0) == 21.0
    assert resolve_supply_temperature_setpoint(control, 10.0, 22.0) == 17.0


def test_duplicate_compensation_reference_temperature_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
            points=(
                _point(21.0, 19.0),
                _point(21.0, 17.0),
            ),
        )


@pytest.mark.parametrize(
    "mode",
    [
        SupplyTemperatureControlMode.OUTDOOR_COMPENSATED,
        SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
    ],
)
def test_compensated_control_requires_two_points(mode):
    with pytest.raises(ValueError, match="at least two points"):
        SupplyTemperatureControl(
            mode=mode,
            points=(_point(21.0, 19.0),),
        )


def test_invalid_minimum_and_maximum_rejected():
    with pytest.raises(ValueError, match="minimum_c must be <= maximum_c"):
        SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.FIXED,
            setpoint_c=18.0,
            minimum_c=20.0,
            maximum_c=19.0,
        )


def test_unsupported_extrapolation_rejected():
    with pytest.raises(ValueError, match="endpoint-clamped"):
        SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
            points=(
                _point(21.0, 19.0),
                _point(23.0, 17.0),
            ),
            extrapolation="linear",
        )


@pytest.mark.parametrize(
    "bad_value",
    [math.nan, math.inf, -math.inf],
)
def test_non_finite_compensation_input_rejected(bad_value):
    control = SupplyTemperatureControl(
        mode=SupplyTemperatureControlMode.EXTRACT_COMPENSATED,
        points=(
            _point(21.0, 19.0),
            _point(23.0, 17.0),
        ),
    )

    with pytest.raises(ValueError, match="extract_temperature_c must be finite"):
        resolve_supply_temperature_setpoint(control, -5.0, bad_value)


def test_control_accepts_string_mode_and_freezes_sorted_points():
    source_points = [
        _point(23.0, 17.0),
        _point(21.0, 19.0),
    ]
    control = SupplyTemperatureControl(
        mode="extract_compensated",
        points=source_points,
    )
    source_points.append(_point(25.0, 16.0))

    assert control.mode is SupplyTemperatureControlMode.EXTRACT_COMPENSATED
    assert tuple(point.reference_temperature_c for point in control.points) == (
        21.0,
        23.0,
    )
