"""Sensible air-handling calculations based on EN 16798-5-1.

The module is intentionally independent of the ISO 52016 zone solver. Supply
temperature control is resolved before the EN 16798-5-1 thermodynamic
calculation so that fixed, scheduled, outdoor-compensated, and
extract-compensated supervisory controls share one explicit and testable
interface.

Only the supply-temperature controller is implemented in this first slice.
Heat recovery, frost protection, coils, fans, and conversion to an ISO 52016
ventilation stream are added separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


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
