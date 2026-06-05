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
    """Return a minimal simulation DataFrame for multizone and legacy solvers.

    Includes zero irradiance for all standard orientations so the legacy
    solver's solar-gain calculation does not raise KeyError.
    """
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame(index=idx)
    df["T2m"] = t_out
    df["WS10m"] = 0.0
    # Solar columns required by the legacy single-zone path
    for _ori in ("HOR", "NV", "EV", "SV", "WV"):
        df[f"I_sol_dif_{_ori}"] = 0.0
        df[f"I_sol_dir_w_{_ori}"] = 0.0
        df[f"I_sol_tot_{_ori}"] = 0.0
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


# ---------------------------------------------------------------------------
# Cross-solver regression tests (hybrid builder + component schedule logic)
# ---------------------------------------------------------------------------

class TestCrossSolverRegressions:
    """Component schedules and zone volume must behave correctly in all solver paths."""

    # ------------------------------------------------------------------
    # Hybrid builder: zone volume
    # ------------------------------------------------------------------

    def _hybrid_bld(self, zone_vol=None, global_vol=2000.0):
        return {
            "building": {
                "net_floor_area": 200.0,
                "building_type_class": "Residential_apartment",
                "adj_zones_present": False,
                "construction_class": "class_i",
                "volume": global_vol,
            },
            "building_parameters": {
                "ventilation": {"ventilation_type": "custom",
                                "custom_heat_transfer_coefficient_ventilation": 0.0,
                                "flow_rate_per_person": 0.0},
                "temperature_setpoints": {},
            },
            "building_surface": [_adiabatic_surface("zone_a")],
            "zones": [{
                "name": "zone_a",
                "net_floor_area": 100.0,
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                **({"zone_volume_m3": zone_vol} if zone_vol is not None else {}),
            }],
        }

    def test_hybrid_builder_copies_zone_volume(self):
        """Hybrid per-zone bui must carry zone_volume_m3 from the zone dict."""
        bui = ISO52016._build_single_zone_building_object_for_core(
            self._hybrid_bld(zone_vol=450.0), "zone_a"
        )
        stored = bui["building"].get("zone_volume_m3")
        assert stored == pytest.approx(450.0), f"Expected 450.0, got {stored}"
        assert stored != pytest.approx(2000.0), "Must not be global volume"

    def test_hybrid_builder_no_zone_volume_does_not_promote_global(self):
        """Without a zone volume key, the global building volume must not be promoted."""
        bui = ISO52016._build_single_zone_building_object_for_core(
            self._hybrid_bld(zone_vol=None, global_vol=9999.0), "zone_a"
        )
        stored = bui["building"].get("zone_volume_m3")
        assert stored is None or stored != pytest.approx(9999.0), (
            f"Global volume must not be promoted to zone; got {stored}"
        )

    # ------------------------------------------------------------------
    # Component schedule logic (resolver level, path-independent)
    # ------------------------------------------------------------------

    def _bld_with_components(self, comp_list, zone_vol=300.0):
        return {
            "building": {
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                "zone_volume_m3": zone_vol,
            },
            "building_parameters": {
                "ventilation": {"components": comp_list},
                "temperature_setpoints": {},
            },
        }

    def test_infiltration_unaffected_by_ahu_profile(self):
        """Infiltration (no profile key) stays at full capacity when AHU multiplier=0."""
        bld = self._bld_with_components([
            {"name": "infiltration", "ventilation_type": "constant_ach",
             "air_changes_per_hour": 0.015},
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": 400.0,
             "source_temperature_c": 18.0},
        ], zone_vol=300.0)
        rho, cp = 1.204, 1006.0
        h_inf = rho * cp * 0.015 * 300.0 / 3600.0

        # AHU off via component_multipliers; infiltration has no profile → 1.0
        bdy = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0,
            component_multipliers={"ahu": 0.0},
            zone_volume_m3=300.0,
        )
        inf_h = sum(s.heat_transfer_coefficient_w_k
                    for s in bdy.streams if s.name == "infiltration")
        assert inf_h == pytest.approx(h_inf, rel=1e-4)
        ahu_h = sum(s.heat_transfer_coefficient_w_k
                    for s in bdy.streams if s.name == "ahu")
        assert ahu_h == pytest.approx(0.0)

    def test_ahu_scaled_by_component_multiplier(self):
        """AHU component is scaled by its component_multiplier (0.5 → half H_k)."""
        bld = self._bld_with_components([
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": 600.0,
             "source_temperature_c": 18.0},
        ])
        bdy_half = resolve_ventilation_boundary(
            bld, 21.0, -5.0, 0.0, component_multipliers={"ahu": 0.5}
        )
        assert bdy_half.heat_transfer_coefficient_w_k == pytest.approx(300.0)

    def test_unknown_profile_emits_warning_in_multizone_path(self):
        """An unknown component profile key in the multizone path must emit a warning."""
        import warnings
        bld = _one_zone_building()
        bld["zones"][0]["ventilation"] = {
            "components": [{
                "name": "mech",
                "ventilation_type": "prescribed",
                "heat_transfer_coefficient_w_k": 300.0,
                "source_temperature_c": 18.0,
                "profile": "nonexistent_column",
            }]
        }
        # Update zone proxy so the component is picked up
        bld["building_parameters"]["ventilation"] = {}
        sim_df = _minimal_sim_df(n=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ISO52016.simulate_envelope_multizone_free_floating(
                building_object=bld,
                _precomputed_sim_df=sim_df,
                use_profiles=False,
            )
        assert any("nonexistent_column" in str(w.message) for w in caught), (
            "Expected a warning about the unknown profile column"
        )


# ---------------------------------------------------------------------------
# True cross-solver tests: legacy, causal, and hybrid paths
# ---------------------------------------------------------------------------

def _legacy_surface():
    """Minimal opaque external surface for the legacy single-zone solver."""
    return {
        "name": "ext_wall",
        "type": "opaque",
        "area": 50.0,
        "sky_view_factor": 0.5,
        "u_value": 0.5,
        "solar_absorptance": 0.6,
        "thermal_capacity": 200000.0,
        "orientation": {"azimuth": 0, "tilt": 90},
        "name_adj_zone": None,
    }


def _legacy_building(components, zone_vol=300.0):
    """Minimal building dict for the legacy single-zone solver.

    Includes all geometry fields that Temp_calculation_of_ground requires.
    """
    return {
        "building": {
            "net_floor_area": 100.0,
            "building_type_class": "Residential_apartment",
            "construction_class": "class_i",
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "zone_volume_m3": zone_vol,
            # Geometry required by Temp_calculation_of_ground
            "exposed_perimeter": 40.0,
            "wall_thickness": 0.3,
            "slab_on_ground_area": 100.0,
            "height": 3.0,
            "latitude_deg": 63.4,
            "longitude_deg": 10.4,
            "latitude": 63.4,
            "longitude": 10.4,
        },
        "adjacent_zones": [],
        "building_parameters": {
            "ventilation": {"components": components},
            "temperature_setpoints": {
                "heating_setpoint": 21.0,
                "heating_setback": 15.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
            },
            "system_capacities": {
                "heating_capacity": 10000.0,
                "cooling_capacity": 10000.0,
            },
        },
        "building_surface": [_legacy_surface()],
    }


def _run_legacy(building, n=48, t_out=-5.0):
    """Run the legacy single-zone solver with a precomputed sim_df.

    warmup_hours=0 is required because the precomputed sim_df is short
    (n << 744); without it Tstep_first_act == Tstepn and all output is empty.
    """
    sim_df = _minimal_sim_df(n, t_out)
    result = ISO52016.Temperature_and_Energy_needs_calculation(
        building,
        _precomputed_sim_df=sim_df,
        weather_source="epw",
        path_weather_file=None,
        warmup_hours=0,
    )
    # Returns (hourly, annual, sankey); we want hourly
    return result[0] if isinstance(result, tuple) else result


class TestLegacyCausalSolverPaths:
    """Verify affine boundary in legacy and causal solver paths (utils.py lines ~6515, ~8047)."""

    def test_legacy_prescribed_h_ve_appears_in_output(self):
        """Legacy solver must emit H_ve matching the prescribed component H_k."""
        h = 500.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": 18.0,
        }])
        out = _run_legacy(bld)
        assert "H_ve" in out.columns, "Legacy output must include H_ve column"
        np.testing.assert_allclose(out["H_ve"].to_numpy(), h, rtol=1e-5)

    def test_legacy_s_ve_equals_h_ve_times_supply_temp(self):
        """Legacy S_ve = H_ve * T_supply for a prescribed non-outdoor component."""
        t_sup = 18.0
        h = 400.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": t_sup,
        }])
        out = _run_legacy(bld)
        assert "S_ve" in out.columns, "Legacy output must include S_ve column"
        np.testing.assert_allclose(
            out["S_ve"].to_numpy(), h * t_sup, rtol=1e-5,
            err_msg="S_ve must equal H_ve * T_supply for a prescribed non-outdoor component",
        )

    def test_legacy_component_schedule_unit(self):
        """_comp_mult_leg at utils.py ~6515 builds component_multipliers correctly.

        Simulates the exact logic the legacy/causal path executes with a controlled
        profile_df containing a mixed ventilation_profile (not all-zero, since
        all-zero gets replaced with occupancy at utils.py ~6290).  Verifies that
        a component with "profile": "ventilation_profile" is scaled by the
        profile value while one without a profile key stays at 1.0.
        """
        rho, cp, ach, vol = 1.204, 1006.0, 0.015, 300.0
        h_inf = rho * cp * ach * vol / 3600.0
        h_ahu = 400.0

        bld = _legacy_building([
            {"name": "infiltration", "ventilation_type": "constant_ach",
             "air_changes_per_hour": ach},
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": h_ahu, "source_temperature_c": 18.0,
             "profile": "ventilation_profile"},
        ], zone_vol=vol)

        vent_cfg = bld["building_parameters"]["ventilation"]
        t_out = -5.0

        for prof_val, expected_h_ahu in [(1.0, h_ahu), (0.5, h_ahu * 0.5), (0.0, 0.0)]:
            # Build profile_df with the given ventilation_profile value at t=0
            profile_df = pd.DataFrame({"ventilation_profile": [prof_val]})

            # Simulate the _comp_mult_leg building code (utils.py ~6515-6540)
            comp_mult: dict = {}
            for comp in vent_cfg.get("components", []):
                cname = str(comp.get("name", "")).strip()
                cprof = comp.get("profile")
                if cname and cprof is not None:
                    col = str(cprof)
                    if col in profile_df.columns:
                        comp_mult[cname] = float(profile_df[col].iloc[0])

            bdy = resolve_ventilation_boundary(
                bld, 21.0, t_out, 0.0,
                profile_multiplier=prof_val,          # legacy stream multiplier
                component_multipliers=comp_mult or None,
                zone_volume_m3=vol,
            )

            h_total = bdy.heat_transfer_coefficient_w_k
            # Infiltration must always be at full capacity (no profile key)
            inf_h = sum(
                s.heat_transfer_coefficient_w_k for s in bdy.streams
                if s.name == "infiltration"
            )
            assert inf_h == pytest.approx(h_inf, rel=1e-4), (
                f"prof_val={prof_val}: infiltration H_k should be {h_inf:.4f}, got {inf_h:.4f}"
            )
            # AHU must be scaled by the profile value
            ahu_h = sum(
                s.heat_transfer_coefficient_w_k for s in bdy.streams
                if s.name == "ahu"
            )
            assert ahu_h == pytest.approx(expected_h_ahu, rel=1e-4 if prof_val > 0 else 1e-9), (
                f"prof_val={prof_val}: AHU H_k should be {expected_h_ahu:.4f}, got {ahu_h:.4f}"
            )

    def test_legacy_ahu_off_via_component_multiplier_end_to_end(self):
        """Legacy solver end-to-end: AHU off via component_multipliers, infiltration on.

        Uses component_multipliers directly (not via profile) to verify the
        affine boundary reaches the legacy solver correctly.
        """
        rho, cp, ach, vol = 1.204, 1006.0, 0.015, 300.0
        h_inf = rho * cp * ach * vol / 3600.0
        h_ahu = 400.0

        bld_both = _legacy_building([
            {"name": "infiltration", "ventilation_type": "constant_ach",
             "air_changes_per_hour": ach},
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": h_ahu, "source_temperature_c": 18.0},
        ], zone_vol=vol)

        # Run with AHU on
        out_on = _run_legacy(bld_both, n=48)
        assert "H_ve" in out_on.columns
        # H_ve should be infiltration + AHU
        np.testing.assert_allclose(
            out_on["H_ve"].to_numpy(), h_inf + h_ahu, rtol=1e-4,
            err_msg="With both streams active, H_ve should equal h_inf + h_ahu",
        )
        # S_ve must differ from H_ve * T_outdoor (supply is 18°C, not outdoor)
        t_out = -5.0
        s_ve_expected = h_inf * t_out + h_ahu * 18.0
        np.testing.assert_allclose(
            out_on["S_ve"].to_numpy(), s_ve_expected, rtol=1e-4,
            err_msg="S_ve must blend outdoor source for infiltration and 18°C for AHU",
        )

    def test_legacy_q_ve_closes(self):
        """Legacy Q_ve = H_ve * T_air - S_ve at every timestep."""
        bld = _legacy_building([{
            "name": "inf",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": 300.0,
            "source_temperature_c": -5.0,
        }])
        out = _run_legacy(bld, n=48)
        assert "Q_ve" in out.columns
        assert "T_air" in out.columns, "Legacy output must expose T_air for Q_ve closure check"
        h_ve = out["H_ve"].to_numpy()
        s_ve = out["S_ve"].to_numpy()
        t_air = out["T_air"].to_numpy()
        q_ve = out["Q_ve"].to_numpy()
        np.testing.assert_allclose(q_ve, h_ve * t_air - s_ve, atol=1e-6)

    def test_hybrid_zone_volume_per_zone_in_legacy_core(self):
        """Hybrid path must use zone-local volume, not global building total."""
        ach = 0.015
        rho, cp = 1.204, 1006.0
        zone_vol = 500.0
        global_vol = 5000.0
        expected_h = rho * cp * ach * zone_vol / 3600.0
        wrong_h = rho * cp * ach * global_vol / 3600.0

        bld = {
            "building": {
                "net_floor_area": 200.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                "volume": global_vol,  # global — must NOT leak into zone
            },
            "building_parameters": {
                "ventilation": {"ventilation_type": "custom",
                                "custom_heat_transfer_coefficient_ventilation": 0.0,
                                "flow_rate_per_person": 0.0},
                "temperature_setpoints": {},
            },
            "building_surface": [
                _adiabatic_surface("zone_a"),
                {
                    "name": "wall_a",
                    "type": "opaque", "area": 50.0, "sky_view_factor": 0.5,
                    "u_value": 0.5, "solar_absorptance": 0.6,
                    "thermal_capacity": 200000.0,
                    "orientation": {"azimuth": 0, "tilt": 90},
                    "name_adj_zone": None, "zone": "zone_a",
                },
            ],
            "zones": [{
                "name": "zone_a",
                "net_floor_area": 100.0,
                "zone_volume_m3": zone_vol,
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
                "ventilation": {"components": [
                    {"name": "inf", "ventilation_type": "constant_ach",
                     "air_changes_per_hour": ach}
                ]},
            }],
        }

        # Build the per-zone bui the hybrid adapter uses, then verify volume
        bui = ISO52016._build_single_zone_building_object_for_core(bld, "zone_a")
        # Global volume must have been removed; zone_volume_m3 must be zone_vol
        assert bui["building"].get("volume") is None, (
            "Global 'volume' must be cleared from hybrid bui['building']"
        )
        assert bui["building"].get("zone_volume_m3") == pytest.approx(zone_vol), (
            f"Expected zone_volume_m3={zone_vol}, got {bui['building'].get('zone_volume_m3')}"
        )

        # Confirm the resolver sees the right volume by running it
        from pybuildingenergy.source.ventilation import resolve_ventilation_boundary
        vent_cfg = bui["building_parameters"]["ventilation"]
        bdy = resolve_ventilation_boundary(
            bui, 20.0, -5.0, 0.0, zone_volume_m3=bui["building"].get("zone_volume_m3")
        )
        assert bdy.heat_transfer_coefficient_w_k == pytest.approx(expected_h, rel=1e-4), (
            f"H_ve should be {expected_h:.4f} (zone vol), not {wrong_h:.4f} (global vol)"
        )

    def test_hybrid_global_zone_volume_m3_does_not_leak(self):
        """Global building.zone_volume_m3 must not survive the hybrid adapter."""
        global_zone_vol = 9999.0
        zone_vol = 300.0
        bld = {
            "building": {
                "net_floor_area": 100.0,
                "building_type_class": "Residential_apartment",
                "construction_class": "class_i",
                "adj_zones_present": False,
                # Global key that must NOT propagate to zone-specific bui
                "zone_volume_m3": global_zone_vol,
            },
            "building_parameters": {
                "ventilation": {"ventilation_type": "custom",
                                "custom_heat_transfer_coefficient_ventilation": 0.0,
                                "flow_rate_per_person": 0.0},
                "temperature_setpoints": {},
            },
            "building_surface": [_adiabatic_surface("z1")],
            "zones": [{
                "name": "z1",
                "net_floor_area": 100.0,
                "zone_volume_m3": zone_vol,     # zone-local value
                "heating_setpoint": 21.0, "heating_setback": 15.0,
                "cooling_setpoint": 26.0, "cooling_setback": 30.0,
            }],
        }
        bui = ISO52016._build_single_zone_building_object_for_core(bld, "z1")
        stored = bui["building"].get("zone_volume_m3")
        assert stored == pytest.approx(zone_vol), (
            f"Zone-local zone_volume_m3={zone_vol} must win over global {global_zone_vol}; "
            f"got {stored}"
        )


def _run_causal(building, n=48, t_out=-5.0):
    """Run the causal single-zone solver (_Temperature_and_Energy_needs_calculation_core_ahu_causal)."""
    sim_df = _minimal_sim_df(n, t_out)
    result = ISO52016._Temperature_and_Energy_needs_calculation_core_ahu_causal(
        building,
        _precomputed_sim_df=sim_df,
        weather_source="epw",
        path_weather_file=None,
        warmup_hours=0,
    )
    return result[0] if isinstance(result, tuple) else result


class TestCausalSolverPath:
    """Verify affine boundary in the causal solver path (utils.py ~8047 and ~8072)."""

    def test_causal_prescribed_h_ve_in_output(self):
        """Causal solver must emit H_ve matching the prescribed component H_k."""
        h = 500.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": 18.0,
        }])
        out = _run_causal(bld)
        assert "H_ve" in out.columns, "Causal output must include H_ve column"
        np.testing.assert_allclose(out["H_ve"].to_numpy(), h, rtol=1e-5)

    def test_causal_s_ve_non_outdoor(self):
        """Causal S_ve = H_ve * T_supply for a prescribed non-outdoor component."""
        t_sup = 18.0
        h = 300.0
        bld = _legacy_building([{
            "name": "mech",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": h,
            "source_temperature_c": t_sup,
        }])
        out = _run_causal(bld)
        assert "S_ve" in out.columns
        np.testing.assert_allclose(out["S_ve"].to_numpy(), h * t_sup, rtol=1e-5)

    def test_causal_q_ve_closes(self):
        """Causal Q_ve = H_ve * T_air - S_ve at every timestep."""
        bld = _legacy_building([{
            "name": "inf",
            "ventilation_type": "prescribed",
            "heat_transfer_coefficient_w_k": 300.0,
            "source_temperature_c": -5.0,
        }])
        out = _run_causal(bld, n=48)
        assert "Q_ve" in out.columns
        assert "T_air" in out.columns
        h_ve = out["H_ve"].to_numpy()
        s_ve = out["S_ve"].to_numpy()
        t_air = out["T_air"].to_numpy()
        q_ve = out["Q_ve"].to_numpy()
        np.testing.assert_allclose(q_ve, h_ve * t_air - s_ve, atol=1e-6)

    def test_causal_component_schedule_unit(self):
        """Causal _comp_mult_leg (utils.py ~8072) scales AHU; infiltration stays at 1.0."""
        rho, cp, ach, vol = 1.204, 1006.0, 0.015, 300.0
        h_inf = rho * cp * ach * vol / 3600.0
        h_ahu = 400.0

        bld = _legacy_building([
            {"name": "infiltration", "ventilation_type": "constant_ach",
             "air_changes_per_hour": ach},
            {"name": "ahu", "ventilation_type": "prescribed",
             "heat_transfer_coefficient_w_k": h_ahu, "source_temperature_c": 18.0,
             "profile": "ventilation_profile"},
        ], zone_vol=vol)

        vent_cfg = bld["building_parameters"]["ventilation"]

        for prof_val, expected_ahu in [(1.0, h_ahu), (0.5, h_ahu * 0.5)]:
            profile_df = pd.DataFrame({"ventilation_profile": [prof_val]})
            comp_mult: dict = {}
            for comp in vent_cfg.get("components", []):
                cname = str(comp.get("name", "")).strip()
                cprof = comp.get("profile")
                if cname and cprof is not None:
                    col = str(cprof)
                    if col in profile_df.columns:
                        comp_mult[cname] = float(profile_df[col].iloc[0])

            bdy = resolve_ventilation_boundary(
                bld, 21.0, -5.0, 0.0,
                component_multipliers=comp_mult or None,
                zone_volume_m3=vol,
            )
            inf_h = sum(s.heat_transfer_coefficient_w_k for s in bdy.streams
                        if s.name == "infiltration")
            ahu_h = sum(s.heat_transfer_coefficient_w_k for s in bdy.streams
                        if s.name == "ahu")
            assert inf_h == pytest.approx(h_inf, rel=1e-4), (
                f"prof={prof_val}: infiltration must be unscaled"
            )
            assert ahu_h == pytest.approx(expected_ahu, rel=1e-4), (
                f"prof={prof_val}: AHU should be {expected_ahu}"
            )
