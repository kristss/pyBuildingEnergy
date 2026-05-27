import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path


# ==============================================================================
#                           FIXTURES
# ==============================================================================

@pytest.fixture
def building_data():
    """Fixture per i dati dell'edificio"""
    return {
        "building": {
            "name": "test-cy",
            "azimuth_relative_to_true_north": 41.8,
            "latitude": 46.49018685497359,
            "longitude": 11.327028776009655,
            "exposed_perimeter": 40,
            "height": 3,
            "wall_thickness": 0.3,
            "n_floors": 1,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "number_adj_zone": 2,
            "net_floor_area": 100,
            "construction_class": "class_i"
        },
        "adjacent_zones": [
            {
                "name": "adj_1",
                "orientation_zone": {"azimuth": 0},
                "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
                "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
                "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
                "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
                'volume': 300,
                'building_type_class': 'Residential_apartment',
                'a_use': 50
            },
            {
                "name": "adj_2",
                "orientation_zone": {"azimuth": 180},
                "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
                "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
                "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
                "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
                'volume': 300,
                'building_type_class': 'Residential_apartment',
                'a_use': 50
            }
        ],
        "building_surface": [
            {
                "name": "Roof surface",
                "type": "opaque",
                "area": 130,
                "sky_view_factor": 1.0,
                "u_value": 2.2,
                "solar_absorptance": 0.4,
                "thermal_capacity": 741500.0,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None
            },
            {
                "name": "Opaque north surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.4,
                "solar_absorptance": 0.4,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 0, "tilt": 90},
                "name_adj_zone": "adj_1"
            },
            {
                "name": "Opaque south surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.4,
                "solar_absorptance": 0.4,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 180, "tilt": 90},
                "name_adj_zone": "adj_2"
            },
            {
                "name": "Opaque east surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.2,
                "solar_absorptance": 0.6,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 90, "tilt": 90},
                "name_adj_zone": None
            },
            {
                "name": "Opaque west surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.2,
                "solar_absorptance": 0.7,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 270, "tilt": 90},
                "name_adj_zone": None
            },
            {
                "name": "Slab to ground",
                "type": "opaque",
                "area": 100,
                "sky_view_factor": 0.0,
                "u_value": 1.6,
                "solar_absorptance": 0.6,
                "thermal_capacity": 405801,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None
            },
            {
                "name": "Transparent east surface",
                "type": "transparent",
                "area": 4,
                "sky_view_factor": 0.5,
                "u_value": 5,
                "g_value": 0.726,
                "height": 2,
                "width": 1,
                "parapet": 1.1,
                "orientation": {"azimuth": 90, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.5,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
                "name_adj_zone": None
            },
            {
                "name": "Transparent west surface",
                "type": "transparent",
                "area": 4,
                "sky_view_factor": 0.5,
                "u_value": 5,
                "g_value": 0.726,
                "height": 2,
                "width": 1,
                "parapet": 1.1,
                "orientation": {"azimuth": 270, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.5,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
                "name_adj_zone": None
            }
        ],
        "units": {
            "area": "m²",
            "u_value": "W/m²K",
            "thermal_capacity": "J/kgK",
            "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
            "tilt": "degrees (0=horizontal, 90=vertical)",
            "internal_gain": "W/m²",
            "internal_gain_profile": "Normalized to 0-1",
            "HVAC_profile": "0: off, 1: on"
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0,
                "heating_setback": 17.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
                "units": "°C"
            },
            "system_capacities": {
                "heating_capacity": 10000000.0,
                "cooling_capacity": 12000000.0,
                "units": "W"
            },
            "airflow_rates": {
                "infiltration_rate": 1.0,
                "units": "ACH (air changes per hour)"
            },
            "internal_gains": [
                {
                    "name": "occupants",
                    "full_load": 4.2,
                    "weekday": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 1.0, 1.0],
                    "weekend": [1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0]
                },
                {
                    "name": "appliances",
                    "full_load": 3,
                    "weekday": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
                    "weekend": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
                },
                {
                    "name": "lighting",
                    "full_load": 3,
                    "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
                    "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
                }
            ],
            "construction": {
                "wall_thickness": 0.3,
                "thermal_bridges": 2,
                "units": "m (for thickness), W/mK (for thermal bridges)"
            },
            "climate_parameters": {
                "coldest_month": 1,
                "units": "1-12 (January-December)"
            },
            "heating_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            },
            "cooling_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            },
            "ventilation_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            }
        }
    }


@pytest.fixture
def hvac_system_config():
    """Fixture per la configurazione del sistema HVAC"""
    return {
        'emitter_type': 'Floor heating',
        'nominal_power': 8,
        'emission_efficiency': 90,
        'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
        'selected_emm_cont_circuit': 0,
        'mixing_valve': True,
        'mixing_valve_delta': 2,
        'heat_losses_recovered': True,
        'distribution_loss_recovery': 90,
        'simplified_approach': 80,
        'distribution_aux_recovery': 80,
        'distribution_aux_power': 30,
        'distribution_loss_coeff': 48,
        'distribution_operation_time': 1,
        'full_load_power': 27,
        'max_monthly_load_factor': 100,
        'tH_gen_i_ON': 1,
        'auxiliary_power_generator': 0,
        'fraction_of_auxiliary_power_generator': 40,
        'generator_circuit': 'independent',
        'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature',
        'gen_outdoor_temp_data': pd.DataFrame({
            "θext_min_gen": [-7],
            "θext_max_gen": [15],
            "θflw_gen_max": [60],
            "θflw_gen_min": [35],
        }, index=["Generator curve"]),
        'speed_control_generator_pump': 'variable',
        'generator_nominal_deltaT': 20,
        'efficiency_model': 'simple',
        'calc_when_QH_positive_only': False,
        'off_compute_mode': 'full',
    }


@pytest.fixture
def output_dir(tmp_path):
    """Fixture per la directory di output temporanea"""
    test_output = tmp_path / "result_test"
    test_output.mkdir()
    return str(test_output)


# ==============================================================================
#                           TESTS
# ==============================================================================

def test_import_package():
    """Test per verificare che il package sia importabile"""
    import pybuildingenergy as pybui
    assert hasattr(pybui, "__version__")


def test_check_heating_system_inputs(hvac_system_config):
    """Test per la validazione degli input del sistema di riscaldamento"""
    import pybuildingenergy as pybui
    
    res = pybui.check_heating_system_inputs(hvac_system_config)
    
    assert "emitter_type" in res
    assert "messages" in res
    assert "config" in res
    assert res["emitter_type"] == "Floor heating 1"


def test_heating_system_calculator(hvac_system_config):
    """Test per il calcolatore del sistema di riscaldamento"""
    import pybuildingenergy as pybui
    
    calc = pybui.HeatingSystemCalculator(hvac_system_config)
    assert calc is not None


def test_heat_pump_system_calculator_heating_cooling_dhw():
    """Test heat pump generation for heating, cooling and DHW demand."""
    import pybuildingenergy as pybui

    heating_map = pd.DataFrame({
        "source_temperature_C": [-7, -7, 2, 2, 7, 7],
        "sink_temperature_C": [35, 55, 35, 55, 35, 55],
        "capacity_kW": [5.0, 4.0, 6.0, 5.0, 7.0, 6.0],
        "cop": [3.2, 2.4, 3.8, 2.8, 4.2, 3.2],
    })
    cooling_map = pd.DataFrame({
        "source_temperature_C": [25, 25, 35, 35],
        "sink_temperature_C": [7, 18, 7, 18],
        "capacity_kW": [5.0, 6.0, 4.0, 5.0],
        "eer": [3.0, 3.6, 2.5, 3.1],
    })
    loads = pd.DataFrame({
        "T_ext": [-5.0, 0.0, 5.0, 12.0, 25.0, 30.0],
        "Q_H_kWh": [4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
        "Q_C_kWh": [0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
        "Q_W_kWh": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }, index=pd.date_range("2026-01-01", periods=6, freq="h"))

    calc = pybui.HeatPumpSystemCalculator({
        "heating_performance_map": heating_map,
        "dhw_performance_map": heating_map,
        "cooling_performance_map": cooling_map,
        "source_type": "air",
        "time_step_hours": 1.0,
        "demand_unit": "kWh",
        "hp_operating_limit_C": 58.0,
        "dhw_target_temperature_C": 55.0,
        "dhw_sink_temperature_C": 55.0,
        "external_auxiliary_power_W": 100.0,
        "standby_power_W": 5.0,
        "heating_storage_loss_kWh_per_day": 0.1,
        "dhw_storage_loss_kWh_per_day": 0.2,
    })

    result = calc.run_timeseries(loads)

    assert result.bins is not None
    assert len(result.bins) > 0
    assert result.summary["QH_gen_out_kWh"] == pytest.approx(10.0)
    assert result.summary["QW_gen_out_kWh"] == pytest.approx(3.0)
    assert result.summary["QC_gen_out_kWh"] == pytest.approx(5.0)
    assert result.summary["E_total_electricity_kWh"] > 0
    assert result.summary["SPF_HW_gen"] > 1.0
    assert result.summary["SEER_C_gen"] > 1.0


def test_emission_system_calculator_heating_cooling_effects():
    """Test EN 15316-2 emission losses and auxiliary electricity."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [-5.0, 0.0, 30.0, 34.0],
            "T_H_int_ini_C": [20.0, 20.0, 20.0, 20.0],
            "T_C_int_ini_C": [26.0, 26.0, 26.0, 26.0],
            "Q_H_kWh": [4.0, 2.0, 0.0, 0.0],
            "Q_C_kWh": [0.0, 0.0, 2.0, 3.0],
            "Q_H_em_out_inc_kWh": [4.2, 2.1, 0.0, 0.0],
            "Q_C_em_out_inc_kWh": [0.0, 0.0, 2.15, 3.2],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )

    calc = pybui.EmissionSystemCalculator(
        {
            "demand_unit": "kWh",
            "heating": {
                "stratification_K": 0.35,
                "control_K": 0.70,
                "hydraulic_balancing_K": 0.10,
                "room_automation_K": -0.50,
                "fan_power_W": 10.0,
                "fan_count": 2.0,
                "convective_fraction": 0.95,
            },
            "cooling": {
                "stratification_K": 0.40,
                "control_K": 0.70,
                "hydraulic_balancing_K": 0.10,
                "room_automation_K": -0.50,
                "fan_power_W": 10.0,
                "fan_count": 2.0,
                "convective_fraction": 0.95,
            },
        }
    )

    result = calc.run_timeseries(loads)

    assert result.timeseries["Q_H_em_in_kWh"].sum() > loads["Q_H_kWh"].sum()
    assert result.timeseries["Q_C_em_in_kWh"].sum() > loads["Q_C_kWh"].sum()
    assert result.summary["QH_em_ls_kWh"] == pytest.approx(0.3)
    assert result.summary["QC_em_ls_kWh"] == pytest.approx(0.35)
    assert result.summary["WH_em_aux_kWh"] > 0
    assert result.summary["WC_em_aux_kWh"] > 0
    assert result.summary["e_H_em_ls_an"] > 1.0
    assert result.summary["e_C_em_ls_an"] > 1.0
    assert pybui.EmissionSystemCalculator is not None


def test_distribution_system_calculator_heating_cooling_dhw():
    """Test EN 15316-3 distribution losses and pump auxiliaries."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [-5.0, 0.0, 30.0, 34.0],
            "T_op": [20.0, 20.0, 26.0, 26.0],
            "Q_H_kWh": [4.0, 2.0, 0.0, 0.0],
            "Q_C_kWh": [0.0, 0.0, 2.0, 3.0],
            "Q_W_kWh": [0.5, 0.5, 0.5, 0.5],
            "theta_H_dis_supply_C": [50.0, 50.0, 50.0, 50.0],
            "theta_H_dis_return_C": [40.0, 40.0, 40.0, 40.0],
            "theta_C_dis_supply_C": [8.0, 8.0, 8.0, 8.0],
            "theta_C_dis_return_C": [14.0, 14.0, 14.0, 14.0],
            "theta_W_dis_hot_C": [58.0, 58.0, 58.0, 58.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )

    common_section = {
        "length_m": 20.0,
        "equivalent_length_m": 0.0,
        "linear_thermal_transmittance_W_mK": 0.30,
        "ambient_temperature_C": 20.0,
        "recoverable": True,
    }
    calc = pybui.DistributionSystemCalculator(
        {
            "demand_unit": "kWh",
            "heating": {
                "pipe_sections": [common_section],
                "supply_temperature_C": 45.0,
                "return_temperature_C": 35.0,
                "nominal_power_kW": 8.0,
                "design_flow_m3_h": 0.7,
                "design_delta_pressure_kPa": 20.0,
                "pump_control_code": 4,
                "eei": 0.23,
            },
            "cooling": {
                "pipe_sections": [{**common_section, "ambient_temperature_C": 26.0}],
                "supply_temperature_C": 7.0,
                "return_temperature_C": 12.0,
                "nominal_power_kW": 6.0,
                "design_flow_m3_h": 0.8,
                "design_delta_pressure_kPa": 20.0,
                "pump_control_code": 3,
                "eei": 0.23,
            },
            "dhw": {
                "pipe_sections": [common_section],
                "dhw_temperature_C": 55.0,
                "dhw_return_deltaT_K": 5.0,
                "design_flow_m3_h": 0.12,
                "design_delta_pressure_kPa": 15.0,
                "pump_control_code": 3,
                "pump_label_power_kW": 0.02,
                "part_load_mode": "constant_when_on",
            },
        }
    )

    result = calc.run_timeseries(loads)

    assert result.summary["QH_dis_ls_kWh"] > 0
    assert result.summary["QC_dis_ls_kWh"] > 0
    assert result.summary["QW_dis_ls_kWh"] > 0
    assert result.summary["WH_dis_aux_kWh"] > 0
    assert result.summary["WC_dis_aux_kWh"] > 0
    assert result.summary["WW_dis_aux_kWh"] > 0
    assert result.summary["QH_dis_in_kWh"] > loads["Q_H_kWh"].sum()
    assert result.summary["QC_dis_in_kWh"] > loads["Q_C_kWh"].sum()
    assert result.summary["QW_dis_in_kWh"] > loads["Q_W_kWh"].sum()
    assert result.summary["QH_dis_rbl_kWh"] > 0
    assert result.summary["QC_dis_rbl_kWh"] < 0
    assert result.timeseries["theta_H_dis_mean_C"].iloc[0] == pytest.approx(45.0)
    assert result.timeseries["theta_C_dis_mean_C"].iloc[0] == pytest.approx(11.0)
    assert result.timeseries["theta_W_dis_mean_C"].iloc[0] == pytest.approx(55.5)
    assert pybui.DistributionSystemCalculator is not None


def test_storage_system_calculator_heating_dhw():
    """Test EN 15316-5 storage losses and pump auxiliaries."""
    import pybuildingenergy as pybui

    loads = pd.DataFrame(
        {
            "T_ext": [-5.0, 0.0, 12.0, 14.0],
            "Q_H_kWh": [4.0, 2.0, 0.0, 0.0],
            "Q_W_kWh": [0.5, 0.5, 0.5, 0.5],
            "theta_H_sto_set_C": [45.0, 45.0, 45.0, 45.0],
            "theta_W_sto_set_C": [55.0, 55.0, 55.0, 55.0],
            "storage_ambient_temperature_C": [20.0, 20.0, 20.0, 20.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )

    calc = pybui.StorageSystemCalculator(
        {
            "demand_unit": "kWh",
            "heating": {
                "storage_volume_l": 80.0,
                "standby_loss_coefficient_W_K": 2.0,
                "set_temperature_C": 45.0,
                "ambient_temperature_C": 20.0,
                "input_pump_power_kW": 0.03,
                "input_pump_flow_m3_h": 0.7,
                "input_pump_deltaT_K": 10.0,
                "thermal_loss_room_fraction": 0.75,
                "auxiliary_to_medium_fraction": 0.25,
            },
            "dhw": {
                "storage_volume_l": 180.0,
                "standby_loss_kWh_per_day_ref": 0.84,
                "standby_set_temperature_ref_C": 55.0,
                "standby_ambient_temperature_ref_C": 20.0,
                "set_temperature_C": 55.0,
                "ambient_temperature_C": 20.0,
                "input_pump_power_kW": 0.025,
                "input_pump_flow_m3_h": 0.4,
                "input_pump_deltaT_K": 5.0,
            },
        }
    )

    result = calc.run_timeseries(loads)

    assert result.summary["QH_sto_ls_kWh"] == pytest.approx(0.2)
    assert result.summary["QW_sto_ls_kWh"] == pytest.approx(0.14)
    assert result.summary["WH_sto_aux_kWh"] > 0
    assert result.summary["WW_sto_aux_kWh"] > 0
    assert result.summary["QH_sto_in_kWh"] > loads["Q_H_kWh"].sum()
    assert result.summary["QW_sto_in_kWh"] > loads["Q_W_kWh"].sum()
    assert result.summary["QH_sto_ls_rbl_kWh"] == pytest.approx(0.15)
    assert result.summary["QH_sto_ls_nrbl_kWh"] == pytest.approx(0.05)
    assert result.timeseries["theta_H_sto_set_C"].iloc[0] == pytest.approx(45.0)
    assert result.timeseries["theta_W_sto_set_C"].iloc[0] == pytest.approx(55.0)
    assert pybui.StorageSystemCalculator is not None


@pytest.mark.parametrize("fix", [True, False])
def test_sanitize_and_validate_bui(building_data, fix):
    """Test per la validazione dei dati dell'edificio"""
    import pybuildingenergy as pybui
    
    bui_result, report = pybui.sanitize_and_validate_BUI(building_data, fix=fix)
    
    assert bui_result is not None
    assert isinstance(report, list)
    
    # Verifica che non ci siano errori critici
    errors = [e for e in report if e["level"] == "ERROR"]
    assert len(errors) == 0, f"Errori trovati: {errors}"


@pytest.mark.slow
def test_iso52016_calculation(building_data, output_dir):
    """Test per il calcolo ISO52016 (può richiedere tempo)"""
    import pybuildingenergy as pybui
    
    # Validazione dati
    bui_checked, issues = pybui.sanitize_and_validate_BUI(building_data, fix=True)
    errors = [e for e in issues if e["level"] == "ERROR"]
    
    assert len(errors) == 0, "Errori nella validazione dei dati"
    
    # Esegui calcolo
    hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        bui_checked,
        weather_source="pvgis"
    )
    
    # Verifica risultati
    assert hourly_sim is not None
    assert annual_results_df is not None
    assert len(hourly_sim) > 0
    assert len(annual_results_df) > 0
    
    # Salva risultati
    hourly_sim.to_csv(os.path.join(output_dir, "hourly_sim_test.csv"))
    annual_results_df.to_csv(os.path.join(output_dir, "annual_results_test.csv"))
    
    # Verifica che i file siano stati creati
    assert os.path.exists(os.path.join(output_dir, "hourly_sim_test.csv"))
    assert os.path.exists(os.path.join(output_dir, "annual_results_test.csv"))


def test_dhw_calculation():
    """Test per il calcolo del fabbisogno di acqua calda sanitaria"""
    import pybuildingenergy as pybui
    
    # Parametri
    teta_W_draw = 42
    teta_W_cold = 11.2
    teta_w_h_ref = 60
    teta_w_c_ref = 13.5
    
    hourly_fractions = pd.DataFrame({
        "Workday": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 20, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0],
        "Weekend": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Holiday": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    
    sum_fractions = pd.DataFrame(hourly_fractions.sum())
    sum_fractions.columns = ["fractions"]
    
    # Genera calendario
    calendar_nation = "Italy"
    italy_calendar = pybui.generate_calendar(calendar_nation, 2023)
    
    n_workdays = sum(italy_calendar['values'] == 'Working')
    n_weekends = sum(italy_calendar['values'] == 'Non-Working')
    n_holidays = sum(italy_calendar['values'] == 'Holiday')
    total_days = italy_calendar.count().values[0]
    
    # Calcolo DHW
    dhw_result = pybui.Volume_and_energy_DHW_calculation(
        n_workdays, n_weekends, n_holidays, sum_fractions, total_days, hourly_fractions,
        teta_W_draw,
        teta_w_c_ref,
        teta_w_h_ref,
        teta_W_cold,
        mode_calc='number_of_units',
        building_type_B3='Residential',
        building_area=142,
        unit_count=10,
        building_type_B5='Dwelling',
        residential_typology='residential_building - simple housing - AVG',
        calculation_method='table',
        year=2015,
        country_calendar=italy_calendar
    )
    
    assert dhw_result is not None
    assert len(dhw_result) > 0


# ==============================================================================
#                           MARKERS
# ==============================================================================

# To run only fast tests: pytest -v -m "not slow"
# To run all tests: pytest -v
