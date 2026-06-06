"""Sensible air-handling calculations based on EN 16798-5-1.

The module is intentionally independent of the ISO 52016 zone solver. Supply
temperature control is resolved before the EN 16798-5-1 thermodynamic
calculation so that fixed, scheduled, outdoor-compensated, and
extract-compensated supervisory controls share one explicit and testable
interface.

Standards note on extract-compensated control
----------------------------------------------
The EXTRACT_COMPENSATED controller computes a requested supply-temperature
setpoint as a piecewise-linear function of zone (extract) air temperature.
This is a supervisory strategy supported by the architecture of EN 16798-5-1
but is NOT enumerated as one of the core controller modes in EN 16798-5-1
Table B.2 (which covers fixed, outdoor-compensated, and load-compensated
strategies). It is implemented here as a generic piecewise-linear controller
that produces the required supply-air temperature for the AHU calculation.

Physical constants
------------------
Standard air at sea level, approximately 20 °C:
  rho = 1.204 kg/m³,  cp = 1006 J/(kg·K)  →  rho*cp = 1211.224 J/(m³·K)

Fan heat placement
------------------
Extract fan heat is added to extract air BEFORE the heat exchanger, increasing
the temperature of air entering the HR core. Supply fan heat is added to
supply air AFTER the heat exchanger and heating coil, just before delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .ventilation import VentilationStream


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_RHO_AIR_KG_M3: float = 1.204
_CP_AIR_J_KG_K: float = 1006.0
_RHO_CP_J_M3_K: float = _RHO_AIR_KG_M3 * _CP_AIR_J_KG_K  # 1211.224 J/(m³·K)


# ---------------------------------------------------------------------------
# Supply-temperature controller
# ---------------------------------------------------------------------------

class SupplyTemperatureControlMode(str, Enum):
    """Available supply-air temperature setpoint strategies."""

    FIXED = "fixed"
    SCHEDULED = "scheduled"
    OUTDOOR_COMPENSATED = "outdoor_compensated"
    EXTRACT_COMPENSATED = "extract_compensated"


@dataclass(frozen=True)
class CompensationPoint:
    """One point on a supply-temperature compensation curve."""

    reference_temperature_c: float
    supply_temperature_c: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_temperature_c):
            raise ValueError("reference_temperature_c must be finite")
        if not math.isfinite(self.supply_temperature_c):
            raise ValueError("supply_temperature_c must be finite")


@dataclass(frozen=True)
class SupplyTemperatureControl:
    """Configuration for a requested AHU supply-temperature setpoint.

    Compensation curves are sorted by reference temperature and use
    piecewise-linear interpolation. Values outside the curve are clamped to
    the nearest endpoint.
    """

    mode: SupplyTemperatureControlMode
    setpoint_c: float | None = None
    points: tuple[CompensationPoint, ...] = ()
    minimum_c: float | None = None
    maximum_c: float | None = None
    extrapolation: str = "clamp"

    def __post_init__(self) -> None:
        try:
            mode = SupplyTemperatureControlMode(self.mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported supply-temperature control mode: {self.mode!r}"
            ) from exc
        object.__setattr__(self, "mode", mode)

        points = tuple(self.points)
        if any(not isinstance(point, CompensationPoint) for point in points):
            raise TypeError("points must contain CompensationPoint instances")
        points = tuple(sorted(points, key=lambda point: point.reference_temperature_c))
        object.__setattr__(self, "points", points)

        references = [point.reference_temperature_c for point in points]
        if len(references) != len(set(references)):
            raise ValueError("Compensation reference temperatures must be unique")

        for name, value in (
            ("setpoint_c", self.setpoint_c),
            ("minimum_c", self.minimum_c),
            ("maximum_c", self.maximum_c),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")

        if (
            self.minimum_c is not None
            and self.maximum_c is not None
            and self.minimum_c > self.maximum_c
        ):
            raise ValueError("minimum_c must be <= maximum_c")
        if self.extrapolation != "clamp":
            raise ValueError("Only endpoint-clamped extrapolation is supported")

        if mode is SupplyTemperatureControlMode.FIXED:
            if self.setpoint_c is None:
                raise ValueError("Fixed control requires setpoint_c")
            if points:
                raise ValueError("Fixed control does not accept compensation points")
        elif mode is SupplyTemperatureControlMode.SCHEDULED:
            if self.setpoint_c is not None:
                raise ValueError("Scheduled control receives its setpoint at runtime")
            if points:
                raise ValueError("Scheduled control does not accept compensation points")
        else:
            if self.setpoint_c is not None:
                raise ValueError("Compensated control does not accept setpoint_c")
            if len(points) < 2:
                raise ValueError("Compensated control requires at least two points")


def resolve_supply_temperature_setpoint(
    control: SupplyTemperatureControl,
    outdoor_temperature_c: float,
    extract_temperature_c: float,
    scheduled_setpoint_c: float | None = None,
) -> float:
    """Return the requested AHU supply-air temperature in degrees Celsius.

    The requested setpoint is distinct from the actual delivered supply
    temperature. Heat recovery, coil capacity, frost control, and disabled
    cooling can make the requested condition unattainable.
    """

    if not isinstance(control, SupplyTemperatureControl):
        raise TypeError("control must be a SupplyTemperatureControl")

    mode = control.mode
    if mode is SupplyTemperatureControlMode.FIXED:
        requested_c = control.setpoint_c
    elif mode is SupplyTemperatureControlMode.SCHEDULED:
        if scheduled_setpoint_c is None:
            raise ValueError("Scheduled control requires scheduled_setpoint_c")
        if not math.isfinite(scheduled_setpoint_c):
            raise ValueError("scheduled_setpoint_c must be finite")
        requested_c = scheduled_setpoint_c
    elif mode is SupplyTemperatureControlMode.OUTDOOR_COMPENSATED:
        requested_c = _interpolate_compensation_curve(
            control.points,
            outdoor_temperature_c,
            "outdoor_temperature_c",
        )
    else:
        requested_c = _interpolate_compensation_curve(
            control.points,
            extract_temperature_c,
            "extract_temperature_c",
        )

    if control.minimum_c is not None:
        requested_c = max(control.minimum_c, requested_c)
    if control.maximum_c is not None:
        requested_c = min(control.maximum_c, requested_c)
    return requested_c


def _interpolate_compensation_curve(
    points: tuple[CompensationPoint, ...],
    reference_temperature_c: float,
    input_name: str,
) -> float:
    if not math.isfinite(reference_temperature_c):
        raise ValueError(f"{input_name} must be finite")

    if reference_temperature_c <= points[0].reference_temperature_c:
        return points[0].supply_temperature_c
    if reference_temperature_c >= points[-1].reference_temperature_c:
        return points[-1].supply_temperature_c

    for lower, upper in zip(points, points[1:]):
        if reference_temperature_c <= upper.reference_temperature_c:
            fraction = (
                (reference_temperature_c - lower.reference_temperature_c)
                / (upper.reference_temperature_c - lower.reference_temperature_c)
            )
            return lower.supply_temperature_c + fraction * (
                upper.supply_temperature_c - lower.supply_temperature_c
            )

    raise RuntimeError("Compensation interpolation failed")


# ---------------------------------------------------------------------------
# AHU thermodynamic model
# ---------------------------------------------------------------------------

class HeatRecoveryControl(str, Enum):
    """Available heat-recovery control strategies."""

    ALWAYS_ON = "always_on"
    MODULATING_BYPASS = "modulating_bypass"


class FrostControlMode(str, Enum):
    """Available frost-protection strategies.

    NONE         — no frost protection; HR efficiency is not reduced.
    EXHAUST_LIMIT — directly reduces the effective HR efficiency so the
                   exhaust leaving the HR core stays at or above
                   ``frost_exhaust_limit_c``.  The continuous exhaust-
                   temperature-limit formulation follows the direct
                   frost-protection approach in EN 16798-5-1 formula (46):
                   heat-recovery efficiency is reduced when frost risk is
                   present.  It does not implement the indirect treatment
                   in formulas (47b) through (49), which calculates
                   frost-protected outdoor-air and limited heat-recovery
                   supply temperatures.  The physical defrost mechanism
                   (bypass, preheat, or recirculation) is a separate
                   configuration dimension not modelled here.
    """

    NONE = "none"
    EXHAUST_LIMIT = "exhaust_limit"


@dataclass(frozen=True)
class SensibleAHUConfig:
    """Configuration for the sensible EN 16798-5-1 AHU timestep model.

    Covers balanced scheduled CAV with sensible heat recovery,
    EXHAUST_LIMIT frost protection, modulating-bypass economizer, and a
    heating coil. Cooling coil support is not yet implemented;
    ``cooling_coil_enabled`` must be False.
    """

    sensible_heat_recovery_efficiency: float
    supply_temperature_control: SupplyTemperatureControl
    heat_recovery_control: HeatRecoveryControl | str
    frost_control: FrostControlMode | str
    heating_coil_max_power_w: float | None
    cooling_coil_enabled: bool
    supply_fan_specific_power_w_per_m3_s: float
    extract_fan_specific_power_w_per_m3_s: float
    supply_fan_heat_fraction_to_air: float
    extract_fan_heat_fraction_to_air: float
    frost_exhaust_limit_c: float = -5.0

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self, "heat_recovery_control",
                HeatRecoveryControl(self.heat_recovery_control),
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported heat_recovery_control: {self.heat_recovery_control!r}"
            ) from exc

        try:
            object.__setattr__(
                self, "frost_control",
                FrostControlMode(self.frost_control),
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported frost_control: {self.frost_control!r}"
            ) from exc

        eta = self.sensible_heat_recovery_efficiency
        if not math.isfinite(eta) or eta < 0.0 or eta > 1.0:
            raise ValueError(
                "sensible_heat_recovery_efficiency must be finite and in [0, 1]"
            )

        if not isinstance(self.supply_temperature_control, SupplyTemperatureControl):
            raise TypeError(
                "supply_temperature_control must be a SupplyTemperatureControl"
            )

        if self.heating_coil_max_power_w is not None:
            if (
                not math.isfinite(self.heating_coil_max_power_w)
                or self.heating_coil_max_power_w < 0.0
            ):
                raise ValueError(
                    "heating_coil_max_power_w must be finite and non-negative when provided"
                )

        for name, value in (
            ("supply_fan_specific_power_w_per_m3_s",
             self.supply_fan_specific_power_w_per_m3_s),
            ("extract_fan_specific_power_w_per_m3_s",
             self.extract_fan_specific_power_w_per_m3_s),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        for name, value in (
            ("supply_fan_heat_fraction_to_air", self.supply_fan_heat_fraction_to_air),
            ("extract_fan_heat_fraction_to_air", self.extract_fan_heat_fraction_to_air),
        ):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

        if not math.isfinite(self.frost_exhaust_limit_c):
            raise ValueError("frost_exhaust_limit_c must be finite")

        if self.cooling_coil_enabled:
            raise NotImplementedError(
                "cooling_coil_enabled=True is not yet implemented; set to False"
            )


@dataclass(frozen=True)
class AHUStepInputs:
    """Time-varying inputs for one AHU timestep."""

    outdoor_temperature_c: float
    extract_temperature_c: float
    required_supply_flow_m3_h: float
    required_extract_flow_m3_h: float
    operation_fraction: float
    timestep_hours: float
    scheduled_setpoint_c: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("outdoor_temperature_c", self.outdoor_temperature_c),
            ("extract_temperature_c", self.extract_temperature_c),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        for name, value in (
            ("required_supply_flow_m3_h", self.required_supply_flow_m3_h),
            ("required_extract_flow_m3_h", self.required_extract_flow_m3_h),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        q_sup = self.required_supply_flow_m3_h
        q_ext = self.required_extract_flow_m3_h
        if q_sup != q_ext:
            larger = max(q_sup, q_ext)
            if larger == 0.0 or abs(q_sup - q_ext) / larger > 1e-6:
                raise ValueError(
                    "required_supply_flow_m3_h and required_extract_flow_m3_h must be "
                    "equal: unequal supply and extract flow is outside the balanced-flow "
                    "scope"
                )

        if (
            not math.isfinite(self.operation_fraction)
            or not (0.0 <= self.operation_fraction <= 1.0)
        ):
            raise ValueError("operation_fraction must be finite and in [0, 1]")

        if not math.isfinite(self.timestep_hours) or self.timestep_hours <= 0.0:
            raise ValueError("timestep_hours must be positive and finite")

        if (
            self.scheduled_setpoint_c is not None
            and not math.isfinite(self.scheduled_setpoint_c)
        ):
            raise ValueError("scheduled_setpoint_c must be finite when provided")


@dataclass(frozen=True)
class AHUStepOutputs:
    """Computed outputs for one AHU timestep.

    Power quantities are in watts, averaged over the timestep. Energy for a
    timestep is power_w * timestep_hours / 1000 kWh.

    requested_supply_temperature_c: setpoint from the supply-temperature controller,
                                    or None when the AHU is off (flow = 0).
    actual_supply_temperature_c:    delivered supply temperature including fan heat.
    heat_recovery_power_w:          rho*cp*q*(T_hr_outlet − T_outdoor).  Equals the
                                    EN 16798-5-1 formula (77) heat-recovery term only
                                    when all of the following hold:
                                    balanced flow, no recirculation, no ODA preheating, and
                                    frost correction inactive.  During active EXHAUST_LIMIT
                                    frost control, the total supply heating relative to
                                    outdoor air may differ from preheat-based defrost
                                    configurations because the effective HR outlet temperature
                                    is determined by a different algorithm.
                                    Positive in heating mode; negative in economizer mode.
    required_heating_coil_power_w:  coil power needed to reach setpoint.
    actual_heating_coil_power_w:    coil power actually delivered (≤ required).
    required_cooling_coil_power_w:  cooling coil power that would be needed to reach the
                                    setpoint but cannot be delivered (cooling_coil_enabled
                                    is not yet implemented).  Non-zero when the requested
                                    supply temperature is below the minimum achievable by
                                    bypass alone.
    fan_electric_power_w:           total fan electrical power (separate from thermal).
    bypass_fraction:                effective fraction of the HR bypassed, derived from
                                    actual vs nominal recovery [0, 1].  Reflects efficiency
                                    reduction rather than a physical bypass damper position.
    frost_protection_required:      True when exhaust temperature would fall below
                                    frost_exhaust_limit_c at nominal HR efficiency.
    """

    requested_supply_temperature_c: float | None
    actual_supply_temperature_c: float
    actual_supply_flow_m3_h: float
    actual_extract_flow_m3_h: float
    required_heating_coil_power_w: float
    actual_heating_coil_power_w: float
    required_cooling_coil_power_w: float
    fan_electric_power_w: float
    heat_recovery_power_w: float
    bypass_fraction: float
    frost_protection_required: bool


def calculate_sensible_ahu_step(
    config: SensibleAHUConfig,
    inputs: AHUStepInputs,
) -> AHUStepOutputs:
    """Calculate one timestep of the sensible EN 16798-5-1 AHU model.

    Calculation sequence
    --------------------
    1.  Scale flows by operation_fraction (scheduled CAV, balanced).  Return
        zero-flow result immediately if AHU is off — avoids requiring a
        scheduled setpoint when the AHU is not running.
    2.  Resolve requested supply-temperature setpoint from config controller.
    3.  Compute fan electricity; add extract fan heat to extract before HR,
        accumulate supply fan heat for addition after HR and coil.
    4.  Apply EXHAUST_LIMIT frost protection: reduce effective HR efficiency
        so the exhaust leaving the HR stays at or above frost_exhaust_limit_c.
    5.  Compute HR outlet temperature from outdoor and frost-limited efficiency.
    6.  Compute fan-adjusted internal coil target: subtract downstream supply fan
        heat from requested setpoint so that delivered temperature equals requested
        after the fan addition.  Apply modulating bypass (clamped to [0, 1]) to
        reach internal target; full bypass yields outdoor air, no bypass yields HR
        outlet.  Compute total bypass_fraction combining frost and modulation.
    7.  Compute heat recovery power (net thermal gain to supply vs outdoor).
    8.  Compute required and actual heating-coil power; raise supply temperature.
        Available coil power = heating_coil_max_power_w * operation_fraction
        (ON/OFF scheduling: coil runs at rated power only during the ON fraction).
    9.  Add supply fan heat to supply after coil.
    10. Return AHUStepOutputs with all quantities separated.

    When operation_fraction is zero or required supply flow is zero, all flow
    and energy outputs are zero and actual supply temperature equals outdoor.
    """
    if not isinstance(config, SensibleAHUConfig):
        raise TypeError("config must be a SensibleAHUConfig")
    if not isinstance(inputs, AHUStepInputs):
        raise TypeError("inputs must be an AHUStepInputs")

    # Step 2 first: Actual flows — needed to detect zero-flow before resolving setpoint
    # so that scheduled control with no setpoint does not raise when the AHU is off.
    op = inputs.operation_fraction
    actual_supply_m3_h = op * inputs.required_supply_flow_m3_h
    actual_extract_m3_h = op * inputs.required_extract_flow_m3_h
    q_sup = actual_supply_m3_h / 3600.0  # m³/s
    q_eta = actual_extract_m3_h / 3600.0  # m³/s

    if q_sup <= 0.0:
        return AHUStepOutputs(
            requested_supply_temperature_c=None,
            actual_supply_temperature_c=inputs.outdoor_temperature_c,
            actual_supply_flow_m3_h=0.0,
            actual_extract_flow_m3_h=0.0,
            required_heating_coil_power_w=0.0,
            actual_heating_coil_power_w=0.0,
            required_cooling_coil_power_w=0.0,
            fan_electric_power_w=0.0,
            heat_recovery_power_w=0.0,
            bypass_fraction=0.0,
            frost_protection_required=False,
        )

    # Step 1: Resolve requested setpoint (only reached when AHU is actually running)
    requested_t_sup = resolve_supply_temperature_setpoint(
        config.supply_temperature_control,
        inputs.outdoor_temperature_c,
        inputs.extract_temperature_c,
        inputs.scheduled_setpoint_c,
    )

    rho_cp_q = _RHO_CP_J_M3_K * q_sup  # W/K for supply stream

    # Step 3: Fan electricity and temperature rises
    p_sup_fan = q_sup * config.supply_fan_specific_power_w_per_m3_s
    p_eta_fan = q_eta * config.extract_fan_specific_power_w_per_m3_s
    fan_electric_power_w = p_sup_fan + p_eta_fan

    # Extract fan heat warms extract entering the HR
    dt_eta_fan = (
        p_eta_fan * config.extract_fan_heat_fraction_to_air / (_RHO_CP_J_M3_K * q_eta)
        if q_eta > 0.0
        else 0.0
    )
    # Supply fan heat is applied after coil (accumulated now, used at step 9)
    dt_sup_fan = p_sup_fan * config.supply_fan_heat_fraction_to_air / rho_cp_q

    t_outdoor = inputs.outdoor_temperature_c
    t_ext_at_hr = inputs.extract_temperature_c + dt_eta_fan
    eta_nom = config.sensible_heat_recovery_efficiency
    dT = t_ext_at_hr - t_outdoor  # K; positive in heating mode

    # Step 4: EXHAUST_LIMIT frost protection — EN 16798-5-1 formula (46)
    # direct effectiveness control. Reduce effective HR efficiency to keep
    # T_EHA >= frost_exhaust_limit_c.
    # The exhaust leaving the HR (EHA) must stay >= frost_exhaust_limit_c.
    # For balanced flow: T_EHA = T_extract_at_hr - eta * dT
    # If T_EHA < limit, reduce eta_eff so T_EHA exactly equals the limit.
    frost_protection_req = False
    eta_eff = eta_nom
    if (
        config.frost_control is FrostControlMode.EXHAUST_LIMIT
        and eta_nom > 0.0
        and abs(dT) > 1e-9
    ):
        t_eha = t_ext_at_hr - eta_nom * dT  # exhaust at HR outlet
        if t_eha < config.frost_exhaust_limit_c:
            frost_protection_req = True
            eta_frost = (t_ext_at_hr - config.frost_exhaust_limit_c) / dT
            eta_eff = max(0.0, min(eta_nom, eta_frost))

    # Step 5: HR outlet temperature (before modulation)
    t_after_hr = t_outdoor + eta_eff * dT

    # Step 6: Modulating bypass and total bypass_fraction.
    # Internal target: subtract downstream supply fan heat so delivered temperature
    # equals requested_t_sup after the fan addition in step 9.
    t_target = requested_t_sup - dt_sup_fan

    # Bypass formula: bp = (T_hr - T_target) / (T_hr - T_outdoor), clamped to [0, 1].
    # bp = 1 → full bypass, delivers T_outdoor.  bp = 0 → no bypass, delivers T_hr.
    # Clamping handles setpoints outside the achievable mixing range: when the
    # setpoint is below T_outdoor (heating mode, no cooling), full bypass delivers
    # the closest attainable temperature; when the setpoint is above T_outdoor
    # (economizer mode), full bypass delivers T_outdoor.
    #
    # Design choice: in economizer mode (T_outdoor > T_extract) we modulate the HR
    # continuously to track the setpoint rather than switching to full outdoor bypass.
    # This keeps supply delivery consistent across all conditions without a mode switch.
    if config.heat_recovery_control is HeatRecoveryControl.MODULATING_BYPASS:
        dT_hr_oda = t_after_hr - t_outdoor  # = eta_eff * dT
        if abs(dT_hr_oda) > 1e-9:
            bp = (t_after_hr - t_target) / dT_hr_oda
            bp = max(0.0, min(1.0, bp))
            t_after_hr = t_outdoor * bp + t_after_hr * (1.0 - bp)

    # Total effective bypass fraction: fraction of supply air bypassing the HR core
    # due to both frost protection (step 4) and setpoint modulation above.
    if eta_nom > 0.0 and abs(dT) > 1e-9:
        bypass_fraction = max(
            0.0, min(1.0, 1.0 - (t_after_hr - t_outdoor) / (eta_nom * dT))
        )
    else:
        bypass_fraction = 0.0

    # Step 7: Heat recovery power (net thermal gain to supply vs outdoor air)
    hr_power_w = rho_cp_q * (t_after_hr - t_outdoor)

    # Unmet cooling: power that would be needed to reach t_target but cannot be
    # delivered because no cooling coil is present.  Positive when bypass has been
    # driven to its limit and the supply is still above the setpoint.
    required_cooling_coil_w = rho_cp_q * max(0.0, t_after_hr - t_target)

    # Step 8: Heating coil targeting t_target so delivered temperature equals
    # requested_t_sup after supply fan heat is added in step 9.
    # For ON/OFF scheduling (operation_fraction < 1), the time-averaged available
    # coil power is coil_max * op: the coil runs at rated power only during the ON
    # fraction, so the timestep-average is proportionally reduced.
    required_heating_w = rho_cp_q * max(0.0, t_target - t_after_hr)
    available_heating_w = (
        config.heating_coil_max_power_w * op
        if config.heating_coil_max_power_w is not None
        else None
    )
    if available_heating_w is None:
        actual_heating_w = required_heating_w
    else:
        actual_heating_w = min(required_heating_w, available_heating_w)

    t_after_coil = t_after_hr + actual_heating_w / rho_cp_q

    # Step 9: Supply fan heat (after HR and coil, before zone delivery)
    t_actual_supply = t_after_coil + dt_sup_fan

    return AHUStepOutputs(
        requested_supply_temperature_c=requested_t_sup,
        actual_supply_temperature_c=t_actual_supply,
        actual_supply_flow_m3_h=actual_supply_m3_h,
        actual_extract_flow_m3_h=actual_extract_m3_h,
        required_heating_coil_power_w=required_heating_w,
        actual_heating_coil_power_w=actual_heating_w,
        required_cooling_coil_power_w=required_cooling_coil_w,
        fan_electric_power_w=fan_electric_power_w,
        heat_recovery_power_w=hr_power_w,
        bypass_fraction=bypass_fraction,
        frost_protection_required=frost_protection_req,
    )


# ---------------------------------------------------------------------------
# ISO 52016-1 zone adapter
# ---------------------------------------------------------------------------

def _opt_float(v) -> "float | None":
    return None if v is None else float(v)


def sensible_ahu_config_from_dict(d: dict) -> SensibleAHUConfig:
    """Build a SensibleAHUConfig from a ventilation component dict.

    Required keys:
      ``sensible_heat_recovery_efficiency`` — float in [0, 1]

    Supply temperature control — choose one mode:

    Fixed (default; use when ``supply_temperature_mode`` is absent or "fixed"):
      ``supply_temperature_setpoint_c``     — fixed setpoint [°C]

    Outdoor-compensated (``supply_temperature_mode = "outdoor_compensated"``):
      ``supply_temperature_points``         — list of [T_oda, T_sup] pairs, e.g.
                                             [[-20, 19], [15, 17]] for ZEBlab
      ``supply_temperature_minimum_c``      — optional clamp [°C]
      ``supply_temperature_maximum_c``      — optional clamp [°C]

    Optional keys and their defaults:
      ``heat_recovery_control``             — "modulating_bypass"
      ``frost_control``                     — "exhaust_limit"
      ``frost_exhaust_limit_c``             — -5.0 [°C]
      ``heating_coil_max_power_w``          — None (unlimited)
      ``cooling_coil_enabled``              — False
      ``supply_fan_specific_power_w_per_m3_s`` — 0.0
      ``extract_fan_specific_power_w_per_m3_s`` — 0.0
      ``supply_fan_heat_fraction_to_air``   — 1.0
      ``extract_fan_heat_fraction_to_air``  — 1.0
    """
    heating_coil_max = d.get("heating_coil_max_power_w")
    if heating_coil_max is not None:
        heating_coil_max = float(heating_coil_max)

    ctrl_mode_str = str(d.get("supply_temperature_mode", "fixed")).strip().lower()
    if ctrl_mode_str in ("fixed", ""):
        raw_setpoint = d.get("supply_temperature_setpoint_c")
        if raw_setpoint is None:
            raise KeyError(
                "fixed supply control requires 'supply_temperature_setpoint_c'"
            )
        supply_ctrl = SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.FIXED,
            setpoint_c=float(raw_setpoint),
            minimum_c=_opt_float(d.get("supply_temperature_minimum_c")),
            maximum_c=_opt_float(d.get("supply_temperature_maximum_c")),
        )
    elif ctrl_mode_str == "outdoor_compensated":
        raw_pts = d.get("supply_temperature_points")
        if not raw_pts:
            raise KeyError(
                "outdoor_compensated control requires 'supply_temperature_points' "
                "(list of [T_oda, T_sup] pairs)"
            )
        points = tuple(
            CompensationPoint(
                reference_temperature_c=float(p[0]),
                supply_temperature_c=float(p[1]),
            )
            for p in raw_pts
        )
        supply_ctrl = SupplyTemperatureControl(
            mode=SupplyTemperatureControlMode.OUTDOOR_COMPENSATED,
            points=points,
            minimum_c=_opt_float(d.get("supply_temperature_minimum_c")),
            maximum_c=_opt_float(d.get("supply_temperature_maximum_c")),
        )
    else:
        raise ValueError(
            f"Unsupported supply_temperature_mode: {ctrl_mode_str!r}. "
            "Supported: 'fixed', 'outdoor_compensated'."
        )

    return SensibleAHUConfig(
        sensible_heat_recovery_efficiency=float(d["sensible_heat_recovery_efficiency"]),
        supply_temperature_control=supply_ctrl,
        heat_recovery_control=str(d.get("heat_recovery_control", "modulating_bypass")),
        frost_control=str(d.get("frost_control", "exhaust_limit")),
        frost_exhaust_limit_c=float(d.get("frost_exhaust_limit_c", -5.0)),
        heating_coil_max_power_w=heating_coil_max,
        cooling_coil_enabled=bool(d.get("cooling_coil_enabled", False)),
        supply_fan_specific_power_w_per_m3_s=float(
            d.get("supply_fan_specific_power_w_per_m3_s", 0.0)
        ),
        extract_fan_specific_power_w_per_m3_s=float(
            d.get("extract_fan_specific_power_w_per_m3_s", 0.0)
        ),
        supply_fan_heat_fraction_to_air=float(
            d.get("supply_fan_heat_fraction_to_air", 1.0)
        ),
        extract_fan_heat_fraction_to_air=float(
            d.get("extract_fan_heat_fraction_to_air", 1.0)
        ),
    )


def ahu_outputs_to_ventilation_stream(
    outputs: AHUStepOutputs,
    name: str = "mechanical_supply",
) -> VentilationStream:
    """Convert AHU step outputs into a VentilationStream for ISO 52016-1 §6.5.10.

    The stream conductance and source temperature couple the AHU into the
    zone affine ventilation boundary:

        H_k = rho_air * cp_air * actual_supply_flow [W/K]
        T_source,k = actual_supply_temperature [°C]

    When the AHU is off (zero flow), H_k = 0 so the stream contributes
    nothing to the zone balance regardless of source temperature.
    """
    h_w_k = _RHO_CP_J_M3_K * outputs.actual_supply_flow_m3_h / 3600.0
    return VentilationStream(
        name=name,
        heat_transfer_coefficient_w_k=h_w_k,
        source_temperature_c=outputs.actual_supply_temperature_c,
        category="supply",
    )
