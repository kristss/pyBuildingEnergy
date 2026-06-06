"""Tests for the solver-independent EN 16798-5-1 sensible AHU model."""

import math

import pytest

from pybuildingenergy.source.ventilation_16798_5_1 import (
    AHUStepInputs,
    AHUStepOutputs,
    CompensationPoint,
    FrostControlMode,
    HeatRecoveryControl,
    SensibleAHUConfig,
    SupplyTemperatureControl,
    SupplyTemperatureControlMode,
    _RHO_CP_J_M3_K,
    calculate_sensible_ahu_step,
    resolve_supply_temperature_setpoint,
)


def _point(reference_c, supply_c):
    return CompensationPoint(reference_c, supply_c)


# ---------------------------------------------------------------------------
# Supply-temperature controller tests (existing)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# AHU model test helpers
# ---------------------------------------------------------------------------

_Q_M3_H = 1350.0  # nominal flow rate used in most tests
_Q_M3_S = _Q_M3_H / 3600.0

_FIXED_18 = SupplyTemperatureControl(
    mode=SupplyTemperatureControlMode.FIXED,
    setpoint_c=18.0,
)


def _cfg(**kw) -> SensibleAHUConfig:
    defaults = dict(
        sensible_heat_recovery_efficiency=0.75,
        supply_temperature_control=_FIXED_18,
        heat_recovery_control="modulating_bypass",
        frost_control="exhaust_limit",
        heating_coil_max_power_w=None,
        cooling_coil_enabled=False,
        supply_fan_specific_power_w_per_m3_s=0.0,
        extract_fan_specific_power_w_per_m3_s=0.0,
        supply_fan_heat_fraction_to_air=1.0,
        extract_fan_heat_fraction_to_air=0.0,
    )
    defaults.update(kw)
    return SensibleAHUConfig(**defaults)


def _inp(**kw) -> AHUStepInputs:
    defaults = dict(
        outdoor_temperature_c=-9.7,
        extract_temperature_c=21.0,
        required_supply_flow_m3_h=_Q_M3_H,
        required_extract_flow_m3_h=_Q_M3_H,
        operation_fraction=1.0,
        timestep_hours=1.0,
    )
    defaults.update(kw)
    return AHUStepInputs(**defaults)


def _rho_cp_q(q_m3_s: float = _Q_M3_S) -> float:
    return _RHO_CP_J_M3_K * q_m3_s


# ---------------------------------------------------------------------------
# SensibleAHUConfig validation
# ---------------------------------------------------------------------------

def test_config_accepts_string_enum_values():
    cfg = _cfg(heat_recovery_control="always_on", frost_control="none")
    assert cfg.heat_recovery_control is HeatRecoveryControl.ALWAYS_ON
    assert cfg.frost_control is FrostControlMode.NONE


def test_config_rejects_unknown_heat_recovery_control():
    with pytest.raises(ValueError, match="heat_recovery_control"):
        _cfg(heat_recovery_control="turbo")


def test_config_rejects_unknown_frost_control():
    with pytest.raises(ValueError, match="frost_control"):
        _cfg(frost_control="preheat")


def test_config_rejects_efficiency_above_one():
    with pytest.raises(ValueError, match="sensible_heat_recovery_efficiency"):
        _cfg(sensible_heat_recovery_efficiency=1.01)


def test_config_rejects_negative_efficiency():
    with pytest.raises(ValueError, match="sensible_heat_recovery_efficiency"):
        _cfg(sensible_heat_recovery_efficiency=-0.1)


def test_config_rejects_non_finite_efficiency():
    with pytest.raises(ValueError):
        _cfg(sensible_heat_recovery_efficiency=math.nan)


def test_config_rejects_negative_fan_power():
    with pytest.raises(ValueError, match="supply_fan_specific_power"):
        _cfg(supply_fan_specific_power_w_per_m3_s=-1.0)


def test_config_rejects_fan_fraction_above_one():
    with pytest.raises(ValueError, match="supply_fan_heat_fraction"):
        _cfg(supply_fan_heat_fraction_to_air=1.1)


def test_config_rejects_zero_heating_coil_max():
    with pytest.raises(ValueError, match="heating_coil_max_power_w"):
        _cfg(heating_coil_max_power_w=0.0)


def test_config_rejects_negative_heating_coil_max():
    with pytest.raises(ValueError, match="heating_coil_max_power_w"):
        _cfg(heating_coil_max_power_w=-100.0)


def test_config_none_heating_coil_max_accepted():
    cfg = _cfg(heating_coil_max_power_w=None)
    assert cfg.heating_coil_max_power_w is None


def test_config_rejects_wrong_supply_temperature_control_type():
    with pytest.raises(TypeError, match="supply_temperature_control"):
        _cfg(supply_temperature_control="fixed:18")


def test_config_rejects_cooling_coil_enabled():
    with pytest.raises(NotImplementedError, match="cooling_coil_enabled"):
        _cfg(cooling_coil_enabled=True)


def test_config_accepts_exhaust_limit_frost_control():
    cfg = _cfg(frost_control="exhaust_limit")
    assert cfg.frost_control is FrostControlMode.EXHAUST_LIMIT


# ---------------------------------------------------------------------------
# AHUStepInputs validation
# ---------------------------------------------------------------------------

def test_inputs_rejects_negative_flow():
    with pytest.raises(ValueError, match="required_supply_flow_m3_h"):
        _inp(required_supply_flow_m3_h=-100.0)


def test_inputs_rejects_operation_fraction_above_one():
    with pytest.raises(ValueError, match="operation_fraction"):
        _inp(operation_fraction=1.5)


def test_inputs_rejects_negative_operation_fraction():
    with pytest.raises(ValueError, match="operation_fraction"):
        _inp(operation_fraction=-0.1)


def test_inputs_rejects_zero_timestep():
    with pytest.raises(ValueError, match="timestep_hours"):
        _inp(timestep_hours=0.0)


def test_inputs_rejects_non_finite_outdoor_temperature():
    with pytest.raises(ValueError, match="outdoor_temperature_c"):
        _inp(outdoor_temperature_c=math.nan)


def test_inputs_rejects_non_finite_scheduled_setpoint():
    with pytest.raises(ValueError, match="scheduled_setpoint_c"):
        _inp(scheduled_setpoint_c=math.inf)


def test_inputs_unbalanced_flows_rejected():
    with pytest.raises(ValueError, match="balanced-flow"):
        _inp(required_extract_flow_m3_h=0.0)  # supply=1350, extract=0 — unbalanced


def test_inputs_both_zero_flows_accepted():
    inp = _inp(required_supply_flow_m3_h=0.0, required_extract_flow_m3_h=0.0)
    assert inp.required_supply_flow_m3_h == 0.0
    assert inp.required_extract_flow_m3_h == 0.0


# ---------------------------------------------------------------------------
# calculate_sensible_ahu_step: type guards
# ---------------------------------------------------------------------------

def test_rejects_non_config():
    with pytest.raises(TypeError, match="config"):
        calculate_sensible_ahu_step("not a config", _inp())


def test_rejects_non_inputs():
    with pytest.raises(TypeError, match="inputs"):
        calculate_sensible_ahu_step(_cfg(), "not inputs")


# ---------------------------------------------------------------------------
# Zero operation
# ---------------------------------------------------------------------------

def test_zero_operation_produces_zero_flow_and_energy():
    out = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.0))
    assert out.actual_supply_flow_m3_h == 0.0
    assert out.actual_extract_flow_m3_h == 0.0
    assert out.fan_electric_power_w == 0.0
    assert out.heat_recovery_power_w == 0.0
    assert out.required_heating_coil_power_w == 0.0
    assert out.actual_heating_coil_power_w == 0.0
    assert out.bypass_fraction == 0.0
    assert not out.frost_control_active


def test_zero_operation_requested_setpoint_is_outdoor_temperature():
    # AHU is off: no setpoint resolution; reported value is outdoor air temperature.
    out = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.0))
    assert out.requested_supply_temperature_c == pytest.approx(-9.7)


def test_scheduled_control_off_ahu_does_not_require_setpoint():
    # Scheduled mode with no setpoint must not raise when AHU is off.
    ctrl = SupplyTemperatureControl(mode="scheduled")
    cfg = _cfg(supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(cfg, _inp(operation_fraction=0.0))
    assert out.actual_supply_flow_m3_h == 0.0
    assert out.actual_heating_coil_power_w == pytest.approx(0.0)


def test_zero_required_flow_equivalent_to_zero_operation():
    out = calculate_sensible_ahu_step(
        _cfg(), _inp(required_supply_flow_m3_h=0.0, required_extract_flow_m3_h=0.0)
    )
    assert out.actual_supply_flow_m3_h == 0.0
    assert out.fan_electric_power_w == 0.0


# ---------------------------------------------------------------------------
# Actual flows scale with operation_fraction
# ---------------------------------------------------------------------------

def test_actual_flows_scale_with_operation_fraction():
    out = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.5))
    assert out.actual_supply_flow_m3_h == pytest.approx(0.5 * _Q_M3_H)
    assert out.actual_extract_flow_m3_h == pytest.approx(0.5 * _Q_M3_H)


# ---------------------------------------------------------------------------
# Heat recovery
# ---------------------------------------------------------------------------

def test_zero_hr_efficiency_no_heat_recovery():
    cfg = _cfg(sensible_heat_recovery_efficiency=0.0)
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-9.7))
    assert out.heat_recovery_power_w == pytest.approx(0.0, abs=1e-9)
    # All heating must come from the coil
    expected_coil = _rho_cp_q() * (18.0 - (-9.7))
    assert out.required_heating_coil_power_w == pytest.approx(expected_coil, rel=1e-6)
    assert out.actual_heating_coil_power_w == pytest.approx(expected_coil, rel=1e-6)


def test_nominal_hr_efficiency_warms_supply():
    # T_outdoor=-9.7, T_extract=21, eta=0.75
    # T_hr = -9.7 + 0.75*(21 - (-9.7)) = -9.7 + 23.025 = 13.325°C
    out = calculate_sensible_ahu_step(_cfg(), _inp())

    t_hr = -9.7 + 0.75 * (21.0 - (-9.7))
    expected_hr = _rho_cp_q() * (t_hr - (-9.7))
    assert out.heat_recovery_power_w == pytest.approx(expected_hr, rel=1e-6)

    expected_coil = _rho_cp_q() * (18.0 - t_hr)
    assert out.required_heating_coil_power_w == pytest.approx(expected_coil, rel=1e-6)
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-9)


def test_sensible_energy_balance_closes():
    """Q_hr + Q_coil = rho*cp*q*(T_supply - T_outdoor)"""
    out = calculate_sensible_ahu_step(_cfg(), _inp())
    total_supply = _rho_cp_q() * (
        out.actual_supply_temperature_c - (-9.7)
    )
    assert total_supply == pytest.approx(
        out.heat_recovery_power_w + out.actual_heating_coil_power_w, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Modulating bypass
# ---------------------------------------------------------------------------

def test_bypass_activates_when_hr_overshoots_setpoint_in_heating_mode():
    # T_outdoor=10, T_extract=22, eta=0.75 → T_hr=10+0.75*12=19°C > setpoint=16°C
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=16.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.75, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    t_hr_full = 10.0 + 0.75 * 12.0  # 19.0°C
    expected_bp = (t_hr_full - 16.0) / (t_hr_full - 10.0)
    assert out.bypass_fraction == pytest.approx(expected_bp, rel=1e-6)
    assert out.actual_supply_temperature_c == pytest.approx(16.0, abs=1e-6)
    assert out.required_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)


def test_bypass_activates_in_economizer_mode_to_avoid_overcooling():
    # T_outdoor=25, T_extract=21, eta=0.75 → T_hr=25+0.75*(21-25)=22°C < setpoint=23°C < 25°C
    # HR overcools; bypass warms supply by mixing in outdoor air
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=23.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.75, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=25.0, extract_temperature_c=21.0)
    )

    t_hr_full = 25.0 + 0.75 * (21.0 - 25.0)  # 22.0°C
    expected_bp = (t_hr_full - 23.0) / (t_hr_full - 25.0)  # (22-23)/(22-25) = 1/3
    assert out.bypass_fraction == pytest.approx(expected_bp, rel=1e-6)
    assert out.actual_supply_temperature_c == pytest.approx(23.0, abs=1e-6)


def test_no_bypass_when_hr_undershoots_setpoint():
    # HR cannot reach setpoint (T_hr < setpoint); heating coil required, no bypass
    out = calculate_sensible_ahu_step(_cfg(), _inp(outdoor_temperature_c=-9.7))
    t_hr = -9.7 + 0.75 * (21.0 - (-9.7))  # 13.325°C < 18°C setpoint
    assert out.bypass_fraction == pytest.approx(0.0, abs=1e-9)
    assert out.heat_recovery_power_w > 0.0
    assert out.required_heating_coil_power_w > 0.0


def test_bypass_zero_when_hr_output_equals_setpoint():
    # T_outdoor=10, T_extract=22, eta=0.75 → T_hr=19°C; setpoint=19°C → no bypass
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=19.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.75, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    assert out.bypass_fraction == pytest.approx(0.0, abs=1e-9)


def test_always_on_hr_does_not_bypass_even_when_setpoint_below_hr_output():
    # HR overshoots setpoint but always_on → no bypass
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=16.0)
    cfg = _cfg(
        heat_recovery_control="always_on",
        supply_temperature_control=ctrl,
        sensible_heat_recovery_efficiency=0.75,
    )
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    assert out.bypass_fraction == pytest.approx(0.0, abs=1e-9)
    t_hr_full = 10.0 + 0.75 * 12.0  # 19.0°C
    assert out.actual_supply_temperature_c == pytest.approx(t_hr_full, abs=1e-9)


# ---------------------------------------------------------------------------
# Frost control
# ---------------------------------------------------------------------------

def test_frost_control_activates_when_exhaust_would_freeze():
    # T_extract=21, T_outdoor=-15, eta=0.75
    # T_EHA_nominal = 21 - 0.75*(21-(-15)) = 21 - 27 = -6°C < limit -5°C → frost
    out = calculate_sensible_ahu_step(
        _cfg(frost_exhaust_limit_c=-5.0),
        _inp(outdoor_temperature_c=-15.0, extract_temperature_c=21.0),
    )
    assert out.frost_control_active


def test_frost_control_reduces_hr_efficiency_to_keep_exhaust_at_limit():
    t_ext, t_oda = 21.0, -15.0
    dT = t_ext - t_oda  # 36 K
    frost_limit = -5.0
    eta_frost = (t_ext - frost_limit) / dT  # 26/36

    out = calculate_sensible_ahu_step(
        _cfg(sensible_heat_recovery_efficiency=0.75, frost_exhaust_limit_c=frost_limit),
        _inp(outdoor_temperature_c=t_oda, extract_temperature_c=t_ext),
    )

    t_hr_expected = t_oda + eta_frost * dT
    expected_hr_power = _rho_cp_q() * (t_hr_expected - t_oda)
    assert out.heat_recovery_power_w == pytest.approx(expected_hr_power, rel=1e-5)


def test_frost_control_not_active_when_exhaust_above_limit():
    # T_extract=21, T_outdoor=-5, eta=0.75
    # T_EHA = 21 - 0.75*(21-(-5)) = 21 - 19.5 = 1.5°C > -5°C → no frost
    out = calculate_sensible_ahu_step(
        _cfg(frost_exhaust_limit_c=-5.0),
        _inp(outdoor_temperature_c=-5.0, extract_temperature_c=21.0),
    )
    assert not out.frost_control_active


def test_frost_mode_none_never_activates():
    cfg = _cfg(frost_control="none", sensible_heat_recovery_efficiency=0.9)
    # Very cold outdoor, would normally trigger frost
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-20.0))
    assert not out.frost_control_active


# ---------------------------------------------------------------------------
# Heating coil
# ---------------------------------------------------------------------------

def test_unlimited_coil_always_reaches_setpoint():
    cfg = _cfg(heating_coil_max_power_w=None)
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-20.0))
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-9)
    assert out.actual_heating_coil_power_w == pytest.approx(
        out.required_heating_coil_power_w, rel=1e-9
    )


def test_capacity_limited_coil_misses_setpoint():
    # Limit to 100 W — far below what is needed for cold outdoor
    cfg = _cfg(heating_coil_max_power_w=100.0)
    out = calculate_sensible_ahu_step(cfg, _inp(outdoor_temperature_c=-9.7))
    assert out.actual_heating_coil_power_w == pytest.approx(100.0, rel=1e-9)
    assert out.actual_heating_coil_power_w < out.required_heating_coil_power_w
    assert out.actual_supply_temperature_c < 18.0


def test_no_heating_needed_when_hr_meets_setpoint_exactly():
    # T_outdoor=10, T_extract=22, eta=0.5 → T_hr=10+0.5*12=16°C; setpoint=16°C
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=16.0)
    cfg = _cfg(sensible_heat_recovery_efficiency=0.5, supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=10.0, extract_temperature_c=22.0)
    )
    assert out.required_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)
    assert out.actual_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)


def test_coil_capacity_scales_with_operation_fraction():
    """Time-averaged available coil power = rated_power * operation_fraction.

    For ON/OFF scheduling the coil runs at rated power during the ON period.
    The supply temperature rise during that period is coil_max / rho_cp_q_nominal,
    identical to full operation — only the time-averaged energy differs.
    """
    # Choose coil_max < required_at_full so the coil is capacity-limited.
    t_oda = -9.7
    t_hr = t_oda + 0.75 * (21.0 - t_oda)  # 13.325°C (no bypass, setpoint > t_hr)
    rho_cp_q_nom = _RHO_CP_J_M3_K * _Q_M3_S
    required_at_full = rho_cp_q_nom * (18.0 - t_hr)  # ~2 233 W
    coil_max = required_at_full * 0.6  # definitely capacity-limited at any op fraction

    cfg = _cfg(heating_coil_max_power_w=coil_max)

    out_full = calculate_sensible_ahu_step(cfg, _inp(operation_fraction=1.0))
    out_half = calculate_sensible_ahu_step(cfg, _inp(operation_fraction=0.5))

    # At full op: actual = coil_max * 1.0
    assert out_full.actual_heating_coil_power_w == pytest.approx(coil_max, rel=1e-9)
    # At half op: time-averaged actual = coil_max * 0.5
    assert out_half.actual_heating_coil_power_w == pytest.approx(coil_max * 0.5, rel=1e-9)
    # Supply temperature during the ON period is the same (coil_max / rho_cp_q_nom
    # temperature rise regardless of duty cycle) — both ops reach the same T_supply.
    assert out_half.actual_supply_temperature_c == pytest.approx(
        out_full.actual_supply_temperature_c, abs=1e-6
    )


# ---------------------------------------------------------------------------
# Cooling disabled — unattainable cold setpoints
# ---------------------------------------------------------------------------

def test_disabled_cooling_full_bypass_when_setpoint_below_outdoor():
    # T_outdoor=5, T_extract=22, setpoint=2°C < T_outdoor — unattainable without cooling.
    # MODULATING_BYPASS: bp = (17.75 - 2) / (17.75 - 5) = 1.235 → clamped to 1.0.
    # Full bypass delivers T_outdoor (5°C); coil has nothing to do.
    ctrl = SupplyTemperatureControl(mode="fixed", setpoint_c=2.0)
    cfg = _cfg(
        sensible_heat_recovery_efficiency=0.75,
        supply_temperature_control=ctrl,
        cooling_coil_enabled=False,
    )
    out = calculate_sensible_ahu_step(
        cfg, _inp(outdoor_temperature_c=5.0, extract_temperature_c=22.0)
    )
    assert out.actual_supply_temperature_c == pytest.approx(5.0, abs=1e-6)
    assert out.bypass_fraction == pytest.approx(1.0, abs=1e-9)
    assert out.required_heating_coil_power_w == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Fan heat and electricity
# ---------------------------------------------------------------------------

def test_extract_fan_heat_warms_extract_before_hr():
    """Extract fan heat increases HR recovery by warming the extract air."""
    sfp = 500.0  # W/(m³/s)
    p_eta = _Q_M3_S * sfp
    dt_eta = p_eta / (_RHO_CP_J_M3_K * _Q_M3_S)

    cfg_with = _cfg(
        extract_fan_specific_power_w_per_m3_s=sfp,
        extract_fan_heat_fraction_to_air=1.0,
        sensible_heat_recovery_efficiency=0.75,
    )
    cfg_without = _cfg(extract_fan_specific_power_w_per_m3_s=0.0)

    out_with = calculate_sensible_ahu_step(cfg_with, _inp())
    out_without = calculate_sensible_ahu_step(cfg_without, _inp())

    assert out_with.heat_recovery_power_w > out_without.heat_recovery_power_w

    # Precise check: T_ext_at_hr increases by dt_eta
    t_ext_at_hr = 21.0 + dt_eta
    dT = t_ext_at_hr - (-9.7)
    t_hr = -9.7 + 0.75 * dT
    expected_hr = _rho_cp_q() * (t_hr - (-9.7))
    assert out_with.heat_recovery_power_w == pytest.approx(expected_hr, rel=1e-5)


def test_extract_fan_heat_zero_fraction_does_not_affect_hr():
    """Extract fan heat fraction=0 means no heat added to extract airstream."""
    sfp = 500.0
    cfg_zero_frac = _cfg(
        extract_fan_specific_power_w_per_m3_s=sfp,
        extract_fan_heat_fraction_to_air=0.0,
    )
    cfg_no_fan = _cfg(extract_fan_specific_power_w_per_m3_s=0.0)

    out_zero_frac = calculate_sensible_ahu_step(cfg_zero_frac, _inp())
    out_no_fan = calculate_sensible_ahu_step(cfg_no_fan, _inp())

    # Same HR power; electricity differs
    assert out_zero_frac.heat_recovery_power_w == pytest.approx(
        out_no_fan.heat_recovery_power_w, rel=1e-9
    )
    assert out_zero_frac.fan_electric_power_w > out_no_fan.fan_electric_power_w


def test_supply_fan_heat_absorbed_by_coil_target_adjustment():
    """Coil targets setpoint minus fan heat rise so delivered equals setpoint."""
    sfp = 500.0
    p_sup = _Q_M3_S * sfp
    dt_sup = p_sup / (_RHO_CP_J_M3_K * _Q_M3_S)

    cfg = _cfg(
        supply_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=1.0,
    )
    out = calculate_sensible_ahu_step(cfg, _inp())

    # Fan is downstream of the coil; the coil targets setpoint - dt_sup so that
    # actual delivered temperature equals the 18°C setpoint after fan heat.
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-5)
    assert out.fan_electric_power_w == pytest.approx(p_sup, rel=1e-9)


def test_supply_fan_zero_heat_fraction_does_not_raise_supply():
    sfp = 500.0
    cfg = _cfg(
        supply_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=0.0,
    )
    out = calculate_sensible_ahu_step(cfg, _inp())
    assert out.actual_supply_temperature_c == pytest.approx(18.0, abs=1e-6)
    assert out.fan_electric_power_w == pytest.approx(_Q_M3_S * sfp, rel=1e-9)


def test_fan_electricity_is_separate_from_thermal_heating():
    sfp = 300.0
    cfg = _cfg(
        supply_fan_specific_power_w_per_m3_s=sfp,
        extract_fan_specific_power_w_per_m3_s=sfp,
        supply_fan_heat_fraction_to_air=0.0,
        extract_fan_heat_fraction_to_air=0.0,
    )
    out = calculate_sensible_ahu_step(cfg, _inp())

    # Fan electricity = both fans, no heat to airstreams
    expected_electric = 2.0 * _Q_M3_S * sfp
    assert out.fan_electric_power_w == pytest.approx(expected_electric, rel=1e-9)

    # Coil power is same as without fans (no thermal interaction)
    out_no_fan = calculate_sensible_ahu_step(_cfg(), _inp())
    assert out.required_heating_coil_power_w == pytest.approx(
        out_no_fan.required_heating_coil_power_w, rel=1e-6
    )


def test_fan_electricity_zero_when_no_fans_configured():
    out = calculate_sensible_ahu_step(_cfg(), _inp())
    assert out.fan_electric_power_w == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Scheduled supply-temperature control in AHU step
# ---------------------------------------------------------------------------

def test_scheduled_control_uses_scheduled_setpoint_c_from_inputs():
    ctrl = SupplyTemperatureControl(mode="scheduled")
    cfg = _cfg(supply_temperature_control=ctrl)
    out = calculate_sensible_ahu_step(cfg, _inp(scheduled_setpoint_c=20.0))
    assert out.requested_supply_temperature_c == pytest.approx(20.0)
    assert out.actual_supply_temperature_c == pytest.approx(20.0, abs=1e-6)


def test_scheduled_control_raises_without_setpoint():
    ctrl = SupplyTemperatureControl(mode="scheduled")
    cfg = _cfg(supply_temperature_control=ctrl)
    with pytest.raises(ValueError, match="requires scheduled_setpoint_c"):
        calculate_sensible_ahu_step(cfg, _inp())


# ---------------------------------------------------------------------------
# Requested setpoint always reported
# ---------------------------------------------------------------------------

def test_requested_setpoint_matches_controller_output():
    ctrl = SupplyTemperatureControl(
        mode="extract_compensated",
        points=(_point(21.0, 19.0), _point(23.0, 17.0)),
    )
    cfg = _cfg(supply_temperature_control=ctrl)
    inp = _inp(extract_temperature_c=22.0)
    out = calculate_sensible_ahu_step(cfg, inp)
    assert out.requested_supply_temperature_c == pytest.approx(18.0)  # midpoint


# ---------------------------------------------------------------------------
# Operation fraction partial load
# ---------------------------------------------------------------------------

def test_partial_load_scales_all_power_outputs():
    out_full = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=1.0))
    out_half = calculate_sensible_ahu_step(_cfg(), _inp(operation_fraction=0.5))

    assert out_half.actual_supply_flow_m3_h == pytest.approx(
        0.5 * out_full.actual_supply_flow_m3_h
    )
    assert out_half.heat_recovery_power_w == pytest.approx(
        0.5 * out_full.heat_recovery_power_w, rel=1e-6
    )
    assert out_half.required_heating_coil_power_w == pytest.approx(
        0.5 * out_full.required_heating_coil_power_w, rel=1e-6
    )
