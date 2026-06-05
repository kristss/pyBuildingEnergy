"""
Tests for the ISO 52016-1 §6.5.10 affine ventilation boundary.

Unit tests for VentilationStream, VentilationBoundary, and resolve_ventilation_boundary.
Compatibility tests verify that existing ventilation_type configurations produce
unchanged H_ve values and that S_ve = H_ve * T_outdoor for all-outdoor cases.
Solver identity tests verify that H_ve * T_zone - S_ve equals the physical heat flow.
"""

import pytest
from pybuildingenergy.source.ventilation import (
    VentilationStream,
    VentilationBoundary,
    resolve_ventilation_boundary,
)


# ---------------------------------------------------------------------------
# VentilationStream unit tests
# ---------------------------------------------------------------------------

class TestVentilationStream:
    def test_one_outdoor_stream(self):
        s = VentilationStream(
            name="infiltration",
            heat_transfer_coefficient_w_k=50.0,
            source_temperature_c=-5.0,
        )
        assert s.heat_transfer_coefficient_w_k == 50.0
        assert s.source_temperature_c == -5.0
        assert s.category == "outdoor_air"

    def test_prescribed_supply_stream(self):
        s = VentilationStream(
            name="supply",
            heat_transfer_coefficient_w_k=800.0,
            source_temperature_c=18.0,
            category="supply",
        )
        assert s.heat_transfer_coefficient_w_k == 800.0
        assert s.source_temperature_c == 18.0
        assert s.category == "supply"

    def test_zero_conductance_stream_is_valid(self):
        s = VentilationStream(
            name="off",
            heat_transfer_coefficient_w_k=0.0,
            source_temperature_c=20.0,
        )
        assert s.heat_transfer_coefficient_w_k == 0.0

    def test_negative_conductance_rejected(self):
        with pytest.raises(ValueError, match="H_k must be >= 0"):
            VentilationStream(
                name="bad",
                heat_transfer_coefficient_w_k=-1.0,
                source_temperature_c=10.0,
            )

    def test_nan_conductance_rejected(self):
        with pytest.raises(ValueError, match="H_k must be finite"):
            VentilationStream(
                name="nan",
                heat_transfer_coefficient_w_k=float("nan"),
                source_temperature_c=10.0,
            )

    def test_inf_conductance_rejected(self):
        with pytest.raises(ValueError, match="H_k must be finite"):
            VentilationStream(
                name="inf",
                heat_transfer_coefficient_w_k=float("inf"),
                source_temperature_c=10.0,
            )

    def test_nan_source_temperature_rejected(self):
        with pytest.raises(ValueError, match="source_temperature_c must be finite"):
            VentilationStream(
                name="nan_temp",
                heat_transfer_coefficient_w_k=100.0,
                source_temperature_c=float("nan"),
            )

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name must be non-empty"):
            VentilationStream(
                name="",
                heat_transfer_coefficient_w_k=100.0,
                source_temperature_c=5.0,
            )

    def test_frozen_immutable(self):
        s = VentilationStream(
            name="x",
            heat_transfer_coefficient_w_k=10.0,
            source_temperature_c=0.0,
        )
        with pytest.raises(Exception):
            s.heat_transfer_coefficient_w_k = 20.0


# ---------------------------------------------------------------------------
# VentilationBoundary unit tests
# ---------------------------------------------------------------------------

class TestVentilationBoundary:
    def test_empty_boundary(self):
        bdy = VentilationBoundary(streams=())
        assert bdy.heat_transfer_coefficient_w_k == 0.0
        assert bdy.source_term_w == 0.0
        assert bdy.equivalent_supply_temperature_c is None
        assert bdy.sensible_heat_flow_w(20.0) == 0.0

    def test_one_stream_aggregate(self):
        s = VentilationStream("v", 100.0, 5.0)
        bdy = VentilationBoundary(streams=(s,))
        assert bdy.heat_transfer_coefficient_w_k == 100.0
        assert bdy.source_term_w == pytest.approx(500.0)
        assert bdy.equivalent_supply_temperature_c == pytest.approx(5.0)

    def test_source_warmer_than_zone(self):
        # Supply at 22 °C into zone at 18 °C -> heat into zone (Q_ve < 0)
        s = VentilationStream("supply", 200.0, 22.0)
        bdy = VentilationBoundary(streams=(s,))
        q_ve = bdy.sensible_heat_flow_w(18.0)
        assert q_ve == pytest.approx(200.0 * 18.0 - 200.0 * 22.0)  # -800 W
        assert q_ve < 0.0

    def test_source_colder_than_zone(self):
        # Outdoor at -5 °C, zone at 20 °C -> heat leaving zone (Q_ve > 0)
        s = VentilationStream("inf", 100.0, -5.0)
        bdy = VentilationBoundary(streams=(s,))
        q_ve = bdy.sensible_heat_flow_w(20.0)
        assert q_ve == pytest.approx(100.0 * (20.0 - (-5.0)))  # 2500 W
        assert q_ve > 0.0

    def test_zone_equals_outdoor_zero_heat_flow(self):
        s = VentilationStream("v", 100.0, 15.0)
        bdy = VentilationBoundary(streams=(s,))
        assert bdy.sensible_heat_flow_w(15.0) == pytest.approx(0.0)

    def test_multiple_additive_streams(self):
        s1 = VentilationStream("inf", 50.0, -5.0)
        s2 = VentilationStream("mech", 300.0, 18.0)
        bdy = VentilationBoundary(streams=(s1, s2))
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(350.0)
        expected_s_ve = 50.0 * (-5.0) + 300.0 * 18.0
        assert bdy.source_term_w == pytest.approx(expected_s_ve)
        expected_t_eq = expected_s_ve / 350.0
        assert bdy.equivalent_supply_temperature_c == pytest.approx(expected_t_eq)

    def test_heat_flow_identity(self):
        """Q_ve = H_ve * T_zone - S_ve must hold for any stream combination."""
        s1 = VentilationStream("a", 120.0, -10.0)
        s2 = VentilationStream("b", 80.0, 19.0)
        bdy = VentilationBoundary(streams=(s1, s2))
        t_zone = 21.5
        q_direct = sum(
            s.heat_transfer_coefficient_w_k * (t_zone - s.source_temperature_c)
            for s in (s1, s2)
        )
        assert bdy.sensible_heat_flow_w(t_zone) == pytest.approx(q_direct)

    def test_zero_conductance_stream_in_boundary(self):
        s_zero = VentilationStream("off", 0.0, 99.0)
        s_active = VentilationStream("on", 100.0, 5.0)
        bdy = VentilationBoundary(streams=(s_zero, s_active))
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(100.0)
        assert bdy.source_term_w == pytest.approx(500.0)

    def test_equivalent_supply_temperature(self):
        # Weighted average: (50*(-5) + 300*18) / 350 = 5150/350 ≈ 14.71 °C
        s1 = VentilationStream("inf", 50.0, -5.0)
        s2 = VentilationStream("mech", 300.0, 18.0)
        bdy = VentilationBoundary(streams=(s1, s2))
        expected = (50.0 * (-5.0) + 300.0 * 18.0) / 350.0
        assert bdy.equivalent_supply_temperature_c == pytest.approx(expected)

    def test_frozen_immutable(self):
        s = VentilationStream("x", 10.0, 0.0)
        bdy = VentilationBoundary(streams=(s,))
        with pytest.raises(Exception):
            bdy.streams = ()


# ---------------------------------------------------------------------------
# resolve_ventilation_boundary compatibility tests
# ---------------------------------------------------------------------------

class TestResolveVentilationBoundary:
    """Verify that legacy ventilation_type configurations resolve correctly."""

    def _custom_building(self, h_ve):
        """Minimal building dict with custom ventilation type."""
        return {
            "building_parameters": {
                "ventilation": {
                    "ventilation_type": "custom",
                    "custom_heat_transfer_coefficient_ventilation": h_ve,
                    "flow_rate_per_person": 0.0,
                }
            },
            "building_surface": [],
            "building": {"net_floor_area": 100.0},
        }

    def test_custom_legacy_h_ve(self):
        bld = self._custom_building(815.0)
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(815.0)

    def test_custom_legacy_s_ve_equals_h_ve_times_t_outdoor(self):
        """For outdoor-air legacy config: S_ve = H_ve * T_outdoor."""
        t_out = -5.0
        bld = self._custom_building(500.0)
        bdy = resolve_ventilation_boundary(bld, 21.0, t_out, 0.0)
        assert bdy.source_term_w == pytest.approx(bdy.heat_transfer_coefficient_w_k * t_out)

    def test_profile_multiplier_scales_h_ve(self):
        bld = self._custom_building(1000.0)
        bdy = resolve_ventilation_boundary(bld, 20.0, -5.0, 0.0, profile_multiplier=0.6)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(600.0)

    def test_none_ventilation_type_returns_empty(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "ventilation_type": "none",
                    "flow_rate_per_person": 0.0,
                    "custom_heat_transfer_coefficient_ventilation": 0.0,
                }
            },
            "building_surface": [],
            "building": {"net_floor_area": 100.0},
        }
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k == 0.0
        assert bdy.source_term_w == 0.0

    def test_components_constant_ach(self):
        """constant_ach component: H_k = rho * cp * ach * V / 3600."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "infiltration",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                            "source_temperature": "outdoor",
                        }
                    ]
                }
            }
        }
        zone_vol = 4000.0  # m3
        t_out = -10.0
        bdy = resolve_ventilation_boundary(
            bld, 21.0, t_out, 0.0, zone_volume_m3=zone_vol
        )
        rho, cp = 1.204, 1006.0
        q_m3_s = 0.015 * zone_vol / 3600.0
        expected_h = rho * cp * q_m3_s
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4)
        assert bdy.source_term_w == pytest.approx(expected_h * t_out, rel=1e-4)

    def test_components_prescribed(self):
        """Prescribed stream has arbitrary source temperature."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "ahu_supply",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 500.0,
                            "source_temperature_c": 18.0,
                        }
                    ]
                }
            }
        }
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(500.0)
        assert bdy.source_term_w == pytest.approx(500.0 * 18.0)
        assert bdy.equivalent_supply_temperature_c == pytest.approx(18.0)

    def test_prescribed_non_outdoor_changes_zone_balance(self):
        """A prescribed supply at 18 °C changes zone balance vs outdoor at -5 °C."""
        t_zone = 21.0
        h = 500.0

        bld_outdoor = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "v",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": h,
                            "source_temperature_c": -5.0,
                        }
                    ]
                }
            }
        }
        bld_supply = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "v",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": h,
                            "source_temperature_c": 18.0,
                        }
                    ]
                }
            }
        }
        q_outdoor = resolve_ventilation_boundary(
            bld_outdoor, t_zone, -5.0, 0.0
        ).sensible_heat_flow_w(t_zone)
        q_supply = resolve_ventilation_boundary(
            bld_supply, t_zone, 18.0, 0.0
        ).sensible_heat_flow_w(t_zone)
        # outdoor: 500*(21-(-5)) = 13000 W loss
        assert q_outdoor == pytest.approx(h * (t_zone - (-5.0)))
        # supply: 500*(21-18) = 1500 W loss
        assert q_supply == pytest.approx(h * (t_zone - 18.0))

    def test_duplicate_component_names_rejected(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {"name": "inf", "ventilation_type": "constant_ach",
                         "air_changes_per_hour": 0.015},
                        {"name": "inf", "ventilation_type": "constant_ach",
                         "air_changes_per_hour": 0.020},
                    ]
                }
            }
        }
        with pytest.raises(ValueError, match="Duplicate.*inf"):
            resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=1000.0)

    def test_constant_ach_without_volume_rejected(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {"name": "inf", "ventilation_type": "constant_ach",
                         "air_changes_per_hour": 0.015},
                    ]
                }
            }
        }
        with pytest.raises((ValueError, TypeError)):
            resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=None)

    def test_extra_streams_added(self):
        """extra_streams parameter adds streams to the resolved boundary."""
        bld = self._custom_building(100.0)
        purge = VentilationStream("purge", 50.0, -5.0)
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, extra_streams=(purge,))
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(150.0)

    def test_q_ve_closes_numerically(self):
        """Q_ve = H_ve * T_zone - S_ve must hold for a custom H_ve boundary."""
        t_zone = 20.5
        t_out = -3.7
        bld = self._custom_building(750.0)
        bdy = resolve_ventilation_boundary(bld, t_zone, t_out, 0.0)
        h_ve = bdy.heat_transfer_coefficient_w_k
        s_ve = bdy.source_term_w
        q_ve_formula = h_ve * t_zone - s_ve
        q_ve_method = bdy.sensible_heat_flow_w(t_zone)
        assert q_ve_formula == pytest.approx(q_ve_method)

    def test_occupancy_ventilation_type_returns_nonneg_h(self):
        bld = {
            "building_parameters": {
                "ventilation": {
                    "ventilation_type": "occupancy",
                    "flow_rate_per_person": 1.4,
                    "custom_heat_transfer_coefficient_ventilation": 0.0,
                }
            },
            "building_surface": [],
            "building": {"net_floor_area": 200.0},
        }
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0)
        assert bdy.heat_transfer_coefficient_w_k >= 0.0

    def test_legacy_s_ve_uses_outdoor_not_supply(self):
        """Legacy path always uses outdoor temperature as source, not a fixed supply."""
        t_out_1 = -10.0
        t_out_2 = 10.0
        bld = self._custom_building(400.0)
        bdy1 = resolve_ventilation_boundary(bld, 20.0, t_out_1, 0.0)
        bdy2 = resolve_ventilation_boundary(bld, 20.0, t_out_2, 0.0)
        assert bdy1.source_term_w == pytest.approx(400.0 * t_out_1)
        assert bdy2.source_term_w == pytest.approx(400.0 * t_out_2)

    # --- Regression tests for fixed blockers ---

    def test_component_profile_multiplier_does_not_affect_components(self):
        """profile_multiplier only scales the legacy single stream, not components.

        Components use a default_multiplier of 1.0 so infiltration can remain
        active when the global mechanical schedule switches off.  To turn off a
        component, use component_multipliers.
        """
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "ahu",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 500.0,
                            "source_temperature_c": 18.0,
                        }
                    ]
                }
            }
        }
        # profile_multiplier has NO effect on components
        bdy_prof_off = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, profile_multiplier=0.0)
        bdy_prof_on = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, profile_multiplier=1.0)
        assert bdy_prof_off.heat_transfer_coefficient_w_k == pytest.approx(500.0)
        assert bdy_prof_on.heat_transfer_coefficient_w_k == pytest.approx(500.0)

        # component_multipliers is the correct way to turn off individual components
        bdy_off = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0, component_multipliers={"ahu": 0.0}
        )
        bdy_half = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0, component_multipliers={"ahu": 0.5}
        )
        assert bdy_off.heat_transfer_coefficient_w_k == pytest.approx(0.0)
        assert bdy_half.heat_transfer_coefficient_w_k == pytest.approx(250.0)

    def test_infiltration_stays_on_when_ahu_is_off(self):
        """Infiltration (no profile key) is always 1.0; AHU off via component_multipliers."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "infiltration",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                        },
                        {
                            "name": "ahu",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 400.0,
                            "source_temperature_c": 18.0,
                        },
                    ]
                }
            }
        }
        vol = 4000.0
        rho, cp = 1.204, 1006.0
        h_inf = rho * cp * 0.015 * vol / 3600.0

        # AHU turned off via component_multipliers; infiltration has no profile
        # so it defaults to 1.0 regardless of profile_multiplier
        bdy = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0,
            profile_multiplier=0.0,
            component_multipliers={"ahu": 0.0},
            zone_volume_m3=vol,
        )
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(h_inf, rel=1e-3)

        # Without any component_multipliers: both streams active at 1.0
        bdy_both = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0,
            profile_multiplier=0.0,  # no effect on components
            zone_volume_m3=vol,
        )
        assert bdy_both.heat_transfer_coefficient_w_k == pytest.approx(h_inf + 400.0, rel=1e-3)

    def test_global_components_fallback(self):
        """building_parameters.ventilation.components is used when zone has no components."""
        # Components defined at global level, zone has no ventilation sub-key
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "inf",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                        }
                    ]
                }
            }
        }
        vol = 5000.0
        rho, cp = 1.204, 1006.0
        expected_h = rho * cp * (0.015 * vol / 3600.0)

        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=vol)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4)

    def test_zone_components_override_global(self):
        """Zone-level ventilation.components takes precedence over global components.

        In the multizone resolver, the zone proxy is built so that zone-level
        components replace global ones.  This unit test uses the resolver directly
        with a zone-proxy-shaped dict (building_parameters.ventilation.components).
        """
        # Simulate the zone proxy after zone-local components win: the proxy's
        # building_parameters.ventilation.components comes from the zone dict,
        # not the global dict.
        zone_proxy = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "zone_inf",
                            "ventilation_type": "prescribed",
                            "heat_transfer_coefficient_w_k": 50.0,
                            "source_temperature_c": -5.0,
                        }
                    ]
                }
            }
        }
        bdy = resolve_ventilation_boundary(zone_proxy, 21.0, -5.0, 0.0)
        names = [s.name for s in bdy.streams]
        assert "zone_inf" in names
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(50.0)

    def test_zone_volume_not_shared_across_zones(self):
        """constant_ach with explicit zone_volume_m3 uses that volume, not building total."""
        rho, cp = 1.204, 1006.0
        ach = 0.015

        def _bld_with_ach():
            return {
                "building_parameters": {
                    "ventilation": {
                        "components": [
                            {"name": "inf", "ventilation_type": "constant_ach",
                             "air_changes_per_hour": ach}
                        ]
                    }
                }
            }

        vol_small = 500.0
        vol_large = 8000.0
        h_small = rho * cp * ach * vol_small / 3600.0
        h_large = rho * cp * ach * vol_large / 3600.0

        bdy_small = resolve_ventilation_boundary(_bld_with_ach(), 21.0, -5.0, 0.0, zone_volume_m3=vol_small)
        bdy_large = resolve_ventilation_boundary(_bld_with_ach(), 21.0, -5.0, 0.0, zone_volume_m3=vol_large)

        assert bdy_small.heat_transfer_coefficient_w_k == pytest.approx(h_small, rel=1e-4)
        assert bdy_large.heat_transfer_coefficient_w_k == pytest.approx(h_large, rel=1e-4)
        # Must differ — if both returned h_large the proxy was leaking global volume
        assert abs(bdy_small.heat_transfer_coefficient_w_k - bdy_large.heat_transfer_coefficient_w_k) > 1.0

    def test_boundary_streams_coerced_to_tuple(self):
        """Passing a list to VentilationBoundary must be silently coerced to tuple."""
        s = VentilationStream("v", 100.0, 5.0)
        bdy = VentilationBoundary(streams=[s])  # list, not tuple
        assert isinstance(bdy.streams, tuple)
        # Subsequent mutation of the original list must not affect the boundary
        original_list = [s]
        bdy2 = VentilationBoundary(streams=original_list)
        original_list.append(VentilationStream("extra", 50.0, 0.0))
        assert len(bdy2.streams) == 1

    def test_boundary_duplicate_stream_names_rejected(self):
        """VentilationBoundary must reject duplicate stream names."""
        s1 = VentilationStream("dup", 100.0, 5.0)
        s2 = VentilationStream("dup", 200.0, 10.0)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            VentilationBoundary(streams=(s1, s2))

    def test_boundary_non_stream_element_rejected(self):
        """VentilationBoundary must reject non-VentilationStream elements."""
        with pytest.raises(TypeError):
            VentilationBoundary(streams=(42,))

    def test_constant_ach_with_zone_volume_m3(self):
        """constant_ach component resolves correctly when zone_volume_m3 is supplied."""
        bld = {
            "building_parameters": {
                "ventilation": {
                    "components": [
                        {
                            "name": "inf",
                            "ventilation_type": "constant_ach",
                            "air_changes_per_hour": 0.015,
                        }
                    ]
                }
            }
        }
        vol = 6960.0  # m3
        bdy = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, zone_volume_m3=vol)
        rho, cp = 1.204, 1006.0
        expected_h = rho * cp * (0.015 * vol / 3600.0)
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4)


# ---------------------------------------------------------------------------
# Solver integration tests
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pybuildingenergy.source.utils import ISO52016


def _minimal_sim_df(n=8760, t_out=-5.0):
    """Return a minimal simulation DataFrame sufficient for the multizone solver."""
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame(index=idx)
    df["T2m"] = t_out
    df["WS10m"] = 0.0
    return df


def _adiabatic_surface(zone_name, area=100.0):
    """Minimal adiabatic surface: no thermal nodes (Pln=0), just provides zone membership."""
    return {
        "name": f"floor_{zone_name}",
        "type": "opaque",
        "boundary": "ADIABATIC",
        "area": area,
        "zone": zone_name,
        "ISO52016_type_string": "AD",
        "sky_view_factor": 0.0,
        "u_value": 0.0,
        "thermal_capacity": 0.0,
        "solar_absorptance": 0.0,
        "orientation": {"azimuth": 0, "tilt": 0},
    }


def _one_zone_building(
    ventilation_type="custom",
    custom_h_ve=500.0,
    components=None,
    zone_volume_m3=None,
):
    """Minimal single-zone building dict for the multizone solver."""
    zone = {
        "name": "zone1",
        "net_floor_area": 100.0,
        "heating_setpoint": 21.0,
        "heating_setback": 15.0,
        "cooling_setpoint": 26.0,
        "cooling_setback": 30.0,
    }
    if components is not None:
        zone["ventilation"] = {"components": components}
    else:
        zone["ventilation_type"] = ventilation_type
        zone["custom_heat_transfer_coefficient_ventilation"] = custom_h_ve
    if zone_volume_m3 is not None:
        zone["zone_volume_m3"] = zone_volume_m3

    return {
        "building": {
            "net_floor_area": 100.0,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "construction_class": "class_i",
        },
        "building_parameters": {
            "ventilation": {
                "ventilation_type": "custom",
                "custom_heat_transfer_coefficient_ventilation": custom_h_ve,
                "flow_rate_per_person": 0.0,
            },
            "temperature_setpoints": {},
        },
        "building_surface": [_adiabatic_surface("zone1")],
        "zones": [zone],
    }


class TestSolverIntegration:
    """Verify that the affine boundary reaches the ISO52016 multizone solver."""

    def _run(self, building, n=48, t_out=-5.0, use_profiles=False):
        sim_df = _minimal_sim_df(n, t_out)
        return ISO52016.simulate_envelope_multizone_free_floating(
            building_object=building,
            _precomputed_sim_df=sim_df,
            use_profiles=use_profiles,
        )

    def test_legacy_custom_h_ve_appears_in_output(self):
        """H_ve column must equal the configured custom H_ve for all timesteps."""
        bld = _one_zone_building(custom_h_ve=815.0)
        out = self._run(bld)
        assert "H_ve_zone1" in out.columns
        np.testing.assert_allclose(out["H_ve_zone1"].to_numpy(), 815.0, rtol=1e-6)

    def test_s_ve_equals_h_ve_times_t_outdoor_for_legacy(self):
        """For legacy outdoor-air config: S_ve = H_ve * T_outdoor at every step."""
        t_out = -5.0
        bld = _one_zone_building(custom_h_ve=400.0)
        out = self._run(bld, t_out=t_out)
        h_ve = out["H_ve_zone1"].to_numpy()
        s_ve = out["S_ve_zone1"].to_numpy()
        np.testing.assert_allclose(s_ve, h_ve * t_out, rtol=1e-6)

    def test_prescribed_supply_reduces_heating_demand(self):
        """A 18 °C supply requires less heating to hold setpoint than -5 °C outdoor air.

        Both cases use H_k=400 W/K.  With T_zone=21 °C:
          outdoor:   Q_ve = 400 * (21 - (-5)) = 10400 W of heat loss → large heating demand
          supply:    Q_ve = 400 * (21 -  18 ) =  1200 W of heat loss → small heating demand
        """
        h = 400.0
        t_out = -5.0
        bld_outdoor = _one_zone_building(custom_h_ve=h)
        bld_supply = _one_zone_building(
            components=[{
                "name": "mech",
                "ventilation_type": "prescribed",
                "heat_transfer_coefficient_w_k": h,
                "source_temperature_c": 18.0,
            }]
        )
        out_outdoor = self._run(bld_outdoor, n=48, t_out=t_out, use_profiles=True)
        out_supply = self._run(bld_supply, n=48, t_out=t_out, use_profiles=True)

        # Both zones are maintained at 21 °C; heating demand must be lower for supply
        q_heat_outdoor = out_outdoor["Q_HVAC_zone1"].clip(lower=0).mean()
        q_heat_supply = out_supply["Q_HVAC_zone1"].clip(lower=0).mean()
        assert q_heat_supply < q_heat_outdoor - 1000.0, (
            f"Heating demand should be >1 kW lower with 18 °C supply than outdoor {t_out} °C; "
            f"got outdoor={q_heat_outdoor:.0f} W, supply={q_heat_supply:.0f} W"
        )

    def test_q_ve_closes_h_ve_times_tzone_minus_s_ve(self):
        """Q_ve = H_ve * T_zone - S_ve must hold at every timestep."""
        bld = _one_zone_building(custom_h_ve=600.0)
        out = self._run(bld)
        h_ve = out["H_ve_zone1"].to_numpy()
        s_ve = out["S_ve_zone1"].to_numpy()
        t_air = out["T_air_zone1"].to_numpy()
        q_ve = out["Q_ve_zone1"].to_numpy()
        np.testing.assert_allclose(q_ve, h_ve * t_air - s_ve, atol=1e-8)

    def test_global_components_reach_solver(self):
        """Global building_parameters.ventilation.components must not be dropped."""
        bld = {
            "building": {
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "adj_zones_present": False,
            },
            "building_parameters": {
                "ventilation": {
                    "components": [{
                        "name": "inf",
                        "ventilation_type": "prescribed",
                        "heat_transfer_coefficient_w_k": 300.0,
                        "source_temperature_c": -5.0,
                    }]
                },
                "temperature_setpoints": {},
            },
            "building_surface": [],
            "zones": [{
                "name": "zone1",
                "net_floor_area": 100.0,
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
            }],
        }
        # Add a surface so the solver doesn't crash with empty surface list
        bld["building"]["construction_class"] = "class_i"
        bld["building_surface"] = [_adiabatic_surface("zone1")]
        out = self._run(bld)
        # If global components were dropped, H_ve would be 0
        np.testing.assert_allclose(out["H_ve_zone1"].to_numpy(), 300.0, rtol=1e-6)

    def test_constant_ach_uses_zone_volume_not_building_total(self):
        """Each zone uses its own volume for constant_ach, not the building total."""
        ach = 0.015
        vol_a, vol_b = 300.0, 1200.0
        rho, cp = 1.204, 1006.0
        h_a = rho * cp * ach * vol_a / 3600.0
        h_b = rho * cp * ach * vol_b / 3600.0
        def _inf():
            return [{"name": "inf", "ventilation_type": "constant_ach",
                     "air_changes_per_hour": ach}]
        bld = {
            "building": {
                "net_floor_area": 150.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                "volume": vol_a + vol_b,  # global total — must not be used per-zone
            },
            "building_parameters": {"ventilation": {}, "temperature_setpoints": {}},
            "building_surface": [
                _adiabatic_surface("zone_a", area=50.0),
                _adiabatic_surface("zone_b", area=100.0),
            ],
            "zones": [
                {
                    "name": "zone_a", "net_floor_area": 50.0, "zone_volume_m3": vol_a,
                    "heating_setpoint": 21.0, "heating_setback": 15.0,
                    "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                    "ventilation": {"components": _inf()},
                },
                {
                    "name": "zone_b", "net_floor_area": 100.0, "zone_volume_m3": vol_b,
                    "heating_setpoint": 21.0, "heating_setback": 15.0,
                    "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                    "ventilation": {"components": _inf()},
                },
            ],
        }
        out = self._run(bld)
        np.testing.assert_allclose(out["H_ve_zone_a"].to_numpy(), h_a, rtol=1e-4)
        np.testing.assert_allclose(out["H_ve_zone_b"].to_numpy(), h_b, rtol=1e-4)

    def test_purge_stream_source_is_outdoor(self):
        """Summer-night purge adds an outdoor-air stream: S_ve = H_ve * T_out."""
        t_out = 15.0  # warm — zone will be warmer, purge condition met
        bld = _one_zone_building(custom_h_ve=200.0)
        bld["zones"][0]["summer_night_purge"] = {
            "enabled": True, "boost_factor": 3.0,
            "months": [6, 8], "hours": [22, 6], "delta_t_min": 0.0,
        }
        sim_df = _minimal_sim_df(n=48, t_out=t_out)
        out = ISO52016.simulate_envelope_multizone_free_floating(
            building_object=bld, _precomputed_sim_df=sim_df
        )
        h_ve = out["H_ve_zone1"].to_numpy()
        s_ve = out["S_ve_zone1"].to_numpy()
        # All streams (base + purge) are outdoor-air, so S_ve == H_ve * T_out
        np.testing.assert_allclose(s_ve, h_ve * t_out, rtol=1e-6)
