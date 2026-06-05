"""
Tests for the ISO 52016-1 §6.5.10 affine ventilation boundary.

Unit tests for VentilationStream, VentilationBoundary, and resolve_ventilation_boundary.
Compatibility tests verify that existing ventilation_type configurations produce
unchanged H_ve values and that S_ve = H_ve * T_outdoor for all-outdoor cases.
Solver identity tests verify that H_ve * T_zone - S_ve equals the physical heat flow.
"""

import math
import pytest
from pybuildingenergy.source.ventilation import (
    VentilationStream,
    VentilationBoundary,
    resolve_ventilation_boundary,
    VentilationInternalGains,
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

    def test_component_profile_multiplier_applied(self):
        """profile_multiplier must scale component H_k (was always returning H=500 W/K)."""
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
        bdy_on = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, profile_multiplier=1.0)
        bdy_off = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, profile_multiplier=0.0)
        bdy_half = resolve_ventilation_boundary(bld, 21.0, -5.0, 0.0, profile_multiplier=0.5)
        assert bdy_on.heat_transfer_coefficient_w_k == pytest.approx(500.0)
        assert bdy_off.heat_transfer_coefficient_w_k == pytest.approx(0.0)
        assert bdy_half.heat_transfer_coefficient_w_k == pytest.approx(250.0)

    def test_component_multipliers_allow_independent_schedules(self):
        """component_multipliers lets infiltration stay on while AHU turns off."""
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
        # AHU off, infiltration on
        bdy = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0,
            profile_multiplier=0.0,
            component_multipliers={"infiltration": 1.0, "ahu": 0.0},
            zone_volume_m3=vol,
        )
        rho, cp = 1.204, 1006.0
        h_inf = rho * cp * 0.015 * vol / 3600.0
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(h_inf, rel=1e-3)
        names = [s.name for s in bdy.streams]
        assert "ahu" not in names or all(
            s.heat_transfer_coefficient_w_k == 0.0
            for s in bdy.streams if s.name == "ahu"
        )

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
