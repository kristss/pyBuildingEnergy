"""Run a heat-pump generation example from ISO52016 and DHW demands.

This script demonstrates the complete sequence:

1. Calculate hourly space heating/cooling needs with ISO52016.
2. Apply EN 15316-2 emitter/control effects and emission losses.
3. Calculate a meaningful hourly DHW need with the DHW module.
4. Apply EN 15316-3 water-based distribution losses and pump auxiliaries.
5. Apply EN 15316-5 heating/DHW storage losses and auxiliaries.
6. Run the EN 15316-4-2 heat-pump bin calculation.
7. Save the intermediate loads, bin balance and summary outputs.

Default weather uses PVGIS for the selected scenario. If network access is not
available, run with ``--weather-source epw --path-weather-file path/to/weather.epw``.
"""

from __future__ import annotations

import argparse
import copy
import html
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pybuildingenergy as pybui  # noqa: E402


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*ChainedAssignmentError.*",
)


SCENARIO_DHW_COUNTRY = {
    "athens": "Greece",
    "bolzano": "Italy",
}


def default_output_dir(
    scenario: str,
    emission_method: str = "en15316-2",
    distribution_method: str = "en15316-3",
    storage_method: str = "en15316-5",
) -> Path:
    suffix = f"heat_pump_15316_4_2_{scenario}"
    if (
        emission_method == "simple"
        and distribution_method == "simple"
        and storage_method == "simple"
    ):
        suffix = f"{suffix}_simple"
    elif (
        emission_method == "en15316-2"
        and distribution_method == "simple"
        and storage_method == "simple"
    ):
        suffix = f"{suffix}_emission_only"
    elif (
        emission_method == "en15316-2"
        and distribution_method == "en15316-3"
        and storage_method == "simple"
    ):
        suffix = f"{suffix}_emission_distribution"
    elif (
        emission_method == "en15316-2"
        and distribution_method == "simple"
        and storage_method == "en15316-5"
    ):
        suffix = f"{suffix}_emission_storage"
    elif (
        emission_method == "simple"
        and distribution_method == "en15316-3"
        and storage_method == "simple"
    ):
        suffix = f"{suffix}_distribution_only"
    elif (
        emission_method == "simple"
        and distribution_method == "simple"
        and storage_method == "en15316-5"
    ):
        suffix = f"{suffix}_storage_only"
    elif (
        emission_method == "simple"
        and distribution_method == "en15316-3"
        and storage_method == "en15316-5"
    ):
        suffix = f"{suffix}_distribution_storage"
    return REPO_ROOT / "examples" / "outputs" / suffix


def example_building(scenario: str = "athens") -> dict:
    """A compact residential building intended to show heating and cooling."""

    if scenario not in SCENARIO_DHW_COUNTRY:
        available = ", ".join(SCENARIO_DHW_COUNTRY)
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {available}")

    weekday_on = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    weekend_cooling = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

    building = {
        "building": {
            "name": "Athens_heat_pump_demo",
            "azimuth_relative_to_true_north": 0,
            "latitude": 37.9888,
            "longitude": 23.7335,
            "exposed_perimeter": 40,
            "height": 6,
            "wall_thickness": 0.35,
            "n_floors": 2,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "net_floor_area": 120,
            "construction_class": "class_i",
            "construction_year": "2010-today",
            "country": "Greece",
        },
        "adjacent_zones": [],
        "building_surface": [
            {
                "name": "Roof surface",
                "type": "opaque",
                "area": 120,
                "sky_view_factor": 1.0,
                "u_value": 0.45,
                "solar_absorptance": 0.5,
                "thermal_capacity": 741500.0,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None,
            },
            {
                "name": "North wall",
                "type": "opaque",
                "area": 60,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.5,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 0, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "South wall",
                "type": "opaque",
                "area": 48,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.65,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 180, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "East wall",
                "type": "opaque",
                "area": 52,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.6,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 90, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "West wall",
                "type": "opaque",
                "area": 52,
                "sky_view_factor": 0.5,
                "u_value": 0.55,
                "solar_absorptance": 0.65,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 270, "tilt": 90},
                "name_adj_zone": None,
            },
            {
                "name": "Slab to ground",
                "type": "opaque",
                "area": 120,
                "sky_view_factor": 0.0,
                "u_value": 0.5,
                "solar_absorptance": 0.6,
                "thermal_capacity": 405801,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None,
            },
            {
                "name": "South glazing",
                "type": "transparent",
                "area": 12,
                "sky_view_factor": 0.5,
                "u_value": 1.7,
                "g_value": 0.55,
                "height": 1.6,
                "width": 7.5,
                "parapet": 0.9,
                "orientation": {"azimuth": 180, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.0,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 0.0},
                "name_adj_zone": None,
            },
            {
                "name": "East glazing",
                "type": "transparent",
                "area": 8,
                "sky_view_factor": 0.5,
                "u_value": 1.7,
                "g_value": 0.55,
                "height": 1.6,
                "width": 5.0,
                "parapet": 0.9,
                "orientation": {"azimuth": 90, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.0,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 0.0},
                "name_adj_zone": None,
            },
            {
                "name": "West glazing",
                "type": "transparent",
                "area": 8,
                "sky_view_factor": 0.5,
                "u_value": 1.7,
                "g_value": 0.55,
                "height": 1.6,
                "width": 5.0,
                "parapet": 0.9,
                "orientation": {"azimuth": 270, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.0,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 0.0},
                "name_adj_zone": None,
            },
        ],
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": 20.0,
                "heating_setback": 17.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
                "units": "C",
            },
            "system_capacities": {
                "heating_capacity": 10000000.0,
                "cooling_capacity": 12000000.0,
                "units": "W",
            },
            "ventilation": {
                "ventilation_type": "custom",
                "flow_rate_per_person": 0.5,
                "units": "W/K when custom ventilation is selected",
                "custom_heat_transfer_coefficient_ventilation": 30.0,
                "info": "ventilation type can be Occupancy, temp_wind or custom.",
            },
            "airflow_rates": {
                "infiltration_rate": 0.7,
                "units": "ACH",
            },
            "internal_gains": [
                {
                    "name": "occupants",
                    "full_load": 5.0,
                    "weekday": [1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.6, 0.6, 0.8, 0.9, 0.9, 0.9, 1, 1],
                    "weekend": [1, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1, 1],
                },
                {
                    "name": "appliances",
                    "full_load": 4.0,
                    "weekday": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.8, 0.8, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.8, 0.8, 1, 1, 1, 0.7, 0.7],
                    "weekend": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.8, 0.8, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.8, 0.8, 1, 1, 1, 0.7, 0.7],
                },
                {
                    "name": "lighting",
                    "full_load": 3.0,
                    "weekday": [0, 0, 0, 0, 0, 0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.15, 0.15],
                    "weekend": [0, 0, 0, 0, 0, 0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.15, 0.15],
                },
            ],
            "construction": {
                "wall_thickness": 0.35,
                "thermal_bridges": 1.5,
                "units": "m and W/K",
            },
            "climate_parameters": {
                "coldest_month": 1,
                "units": "1-12",
            },
            "heating_profile": {"weekday": weekday_on, "weekend": weekday_on},
            "cooling_profile": {"weekday": weekday_on, "weekend": weekend_cooling},
            "ventilation_profile": {"weekday": weekday_on, "weekend": weekend_cooling},
        },
    }

    if scenario == "bolzano":
        building["building"].update(
            {
                "name": "Bolzano_heat_pump_demo",
                "latitude": 46.4983,
                "longitude": 11.3548,
                "construction_year": "1991-2005",
                "construction_class": "class_e",
                "country": "Italy",
            }
        )
        building["building_parameters"]["airflow_rates"]["infiltration_rate"] = 0.30
        building["building_parameters"]["ventilation"]["custom_heat_transfer_coefficient_ventilation"] = 20.0
        building["building_parameters"]["construction"]["thermal_bridges"] = 1.25

        surfaces = {surface["name"]: surface for surface in building["building_surface"]}
        surfaces["Roof surface"]["u_value"] = 0.22
        surfaces["Roof surface"]["solar_absorptance"] = 0.65
        surfaces["North wall"]["u_value"] = 0.30
        surfaces["South wall"]["u_value"] = 0.30
        surfaces["East wall"]["u_value"] = 0.30
        surfaces["West wall"]["u_value"] = 0.30
        surfaces["Slab to ground"]["u_value"] = 0.32

        for name in ["South glazing", "East glazing", "West glazing"]:
            surfaces[name]["u_value"] = 1.2
            surfaces[name]["g_value"] = 0.58

        surfaces["South glazing"]["area"] = 18
        surfaces["South glazing"]["width"] = 11.25
        surfaces["East glazing"]["area"] = 10
        surfaces["East glazing"]["width"] = 6.25
        surfaces["West glazing"]["area"] = 12
        surfaces["West glazing"]["width"] = 7.5

    return building


def run_iso52016(building: dict, weather_source: str, path_weather_file: str | None) -> pd.DataFrame:
    """Run ISO52016 and return the hourly load table."""

    checked, issues = pybui.sanitize_and_validate_BUI(building, fix=True)
    errors = [issue for issue in issues if issue["level"] == "ERROR"]
    if errors:
        details = "\n".join(f"- {item['path']}: {item['msg']}" for item in errors)
        raise RuntimeError(f"The example building is not valid:\n{details}")

    kwargs = {"weather_source": weather_source}
    if weather_source == "epw":
        if not path_weather_file:
            raise ValueError("--path-weather-file is required when --weather-source epw.")
        kwargs["path_weather_file"] = path_weather_file

    result = pybui.ISO52016.Temperature_and_Energy_needs_calculation(checked, **kwargs)
    if not isinstance(result, tuple) or len(result) < 1:
        raise RuntimeError("ISO52016 did not return an hourly result table.")
    return result[0].copy()


def make_dhw_profile(index: pd.DatetimeIndex, building_area: float, country: str) -> pd.Series:
    """Calculate hourly DHW needs and align them to an ISO52016 hourly index."""

    hourly_fractions = pd.DataFrame(
        {
            "Workday": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 20, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0],
            "Weekend": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Holiday": [0, 0, 0, 0, 0, 0, 0, 0, 3, 8, 8, 8, 15, 10, 10, 10, 8, 5, 0, 0, 0, 0, 0, 0],
        }
    )
    sum_fractions = pd.DataFrame(hourly_fractions.sum(), columns=["fractions"])

    annual_profiles = []
    for year in sorted(set(index.year)):
        calendar = pybui.generate_calendar(country, int(year))
        n_workdays = int((calendar["values"] == "Working").sum())
        n_weekends = int((calendar["values"] == "Non-Working").sum())
        n_holidays = int((calendar["values"] == "Holiday").sum())
        total_days = int(calendar["values"].count())

        dhw_result = pybui.Volume_and_energy_DHW_calculation(
            n_workdays,
            n_weekends,
            n_holidays,
            sum_fractions,
            total_days,
            hourly_fractions,
            42.0,
            13.5,
            60.0,
            11.2,
            mode_calc="number_of_units",
            building_type_B3="Residential",
            building_area=building_area,
            unit_count=4,
            building_type_B5="Dwelling",
            residential_typology="residential_building - simple housing - AVG",
            calculation_method="table",
            year=int(year),
            country_calendar=calendar,
        )
        values = pd.Series(dhw_result[7], name="Q_W_kWh")
        year_index = pd.date_range(f"{year}-01-01 00:00:00", periods=len(values), freq="h")
        values.index = year_index
        annual_profiles.append(values)

    profile = pd.concat(annual_profiles).sort_index()
    aligned = profile.reindex(index)
    if aligned.isna().any():
        aligned = aligned.ffill().bfill()
    return aligned.astype(float)


def heat_pump_maps(scenario: str = "athens") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Example product maps. Replace with manufacturer/test data for real studies."""

    if scenario == "bolzano":
        heating_capacity_base = 8.4
        heating_capacity_source_factor = 0.13
        heating_capacity_sink_factor = 0.055
        heating_cop_base = 4.05
        cooling_capacity_base = 6.2
        cooling_capacity_source_factor = 0.07
        cooling_capacity_sink_factor = 0.035
        cooling_eer_base = 3.75
    else:
        heating_capacity_base = 9.0
        heating_capacity_source_factor = 0.12
        heating_capacity_sink_factor = 0.05
        heating_cop_base = 4.1
        cooling_capacity_base = 8.0
        cooling_capacity_source_factor = 0.08
        cooling_capacity_sink_factor = 0.04
        cooling_eer_base = 3.9

    heating_rows = []
    for source in [-15, -7, 2, 7, 12, 20]:
        for sink in [35, 45, 55]:
            cop = heating_cop_base + 0.055 * source - 0.045 * (sink - 35)
            capacity = (
                heating_capacity_base
                + heating_capacity_source_factor * source
                - heating_capacity_sink_factor * (sink - 35)
            )
            heating_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.5),
                    "cop": max(cop, 1.6),
                }
            )

    cooling_rows = []
    for source in [20, 25, 30, 35, 40]:
        for sink in [7, 12, 18]:
            eer = cooling_eer_base - 0.055 * (source - 25) + 0.045 * (sink - 7)
            capacity = (
                cooling_capacity_base
                - cooling_capacity_source_factor * (source - 25)
                + cooling_capacity_sink_factor * (sink - 7)
            )
            cooling_rows.append(
                {
                    "source_temperature_C": source,
                    "sink_temperature_C": sink,
                    "capacity_kW": max(capacity, 2.5),
                    "eer": max(eer, 1.7),
                }
            )

    return pd.DataFrame(heating_rows), pd.DataFrame(cooling_rows)


def heat_pump_config(
    scenario: str,
    heating_map: pd.DataFrame,
    cooling_map: pd.DataFrame,
    include_internal_storage_losses: bool = True,
) -> dict:
    """Scenario-specific generator assumptions for the example heat pump."""

    config = {
        "heating_performance_map": heating_map,
        "dhw_performance_map": heating_map,
        "cooling_performance_map": cooling_map,
        "source_type": "air",
        "demand_unit": "kWh",
        "bin_width_C": 1.0,
        "design_outdoor_temperature_C": -3.0,
        "heating_cutoff_temperature_C": 16.0,
        "heating_sink_temp_at_design_C": 45.0,
        "heating_sink_temp_at_cutoff_C": 30.0,
        "dhw_target_temperature_C": 60.0,
        "dhw_sink_temperature_C": 55.0,
        "dhw_cold_water_temperature_C": 11.2,
        "cooling_sink_temperature_C": 7.0,
        "hp_operating_limit_C": 55.0,
        "backup_mode": "parallel",
        "heating_backup_efficiency": 1.0,
        "dhw_backup_efficiency": 1.0,
        "external_auxiliary_power_W": 120.0,
        "standby_power_W": 8.0,
        "heating_storage_loss_kWh_per_day": 0.15,
        "dhw_storage_loss_kWh_per_day": 0.90,
        "storage_test_deltaT_K": 45.0,
        "storage_ambient_temperature_C": 20.0,
    }

    if scenario == "bolzano":
        config.update(
            {
                "design_outdoor_temperature_C": -7.0,
                "heating_sink_temp_at_design_C": 45.0,
                "heating_sink_temp_at_cutoff_C": 30.0,
                "dhw_cold_water_temperature_C": 9.5,
                "external_auxiliary_power_W": 130.0,
                "heating_storage_loss_kWh_per_day": 0.20,
                "dhw_storage_loss_kWh_per_day": 0.95,
            }
        )

    if not include_internal_storage_losses:
        config["heating_storage_loss_kWh_per_day"] = 0.0
        config["dhw_storage_loss_kWh_per_day"] = 0.0

    return config


def emission_system_config(scenario: str = "athens") -> dict:
    """Scenario-specific EN 15316-2 emission assumptions.

    The defaults represent water-based fan-coil emission with PI room control,
    balanced hydraulics and stand-alone room automation. Values are taken from
    EN 15316-2:2017 Annex B default tables where available.
    """

    config = {
        "demand_unit": "kWh",
        "cooling_solar_gain_temperature_C": 8.0,
        "heating": {
            "stratification_K": 0.35,
            "control_K": 0.70,
            "radiation_K": 0.0,
            "hydraulic_balancing_K": 0.10,
            "room_automation_K": -0.50,
            "embedded_K": 0.0,
            "nominal_power_kW": 10.0,
            "fan_power_W": 10.0,
            "fan_count": 4.0,
            "control_power_W": 0.1,
            "control_count": 4.0,
            "convective_fraction": 0.95,
        },
        "cooling": {
            "stratification_K": 0.40,
            "control_K": 0.70,
            "radiation_K": 0.0,
            "hydraulic_balancing_K": 0.10,
            "room_automation_K": -0.50,
            "embedded_K": 0.0,
            "nominal_power_kW": 8.0,
            "fan_power_W": 10.0,
            "fan_count": 4.0,
            "control_power_W": 0.1,
            "control_count": 4.0,
            "convective_fraction": 0.95,
        },
    }

    if scenario == "bolzano":
        config["heating"]["nominal_power_kW"] = 11.0
        config["cooling"]["nominal_power_kW"] = 6.5

    return config


def building_with_emission_setpoints(
    building: dict,
    emission_calc: pybui.EmissionSystemCalculator,
) -> dict:
    """Return a copy of the building with EN 15316-2 equivalent setpoints."""

    adjusted = copy.deepcopy(building)
    setpoints = adjusted["building_parameters"]["temperature_setpoints"]
    h_delta = emission_calc.temperature_increase_K("H")
    c_delta = emission_calc.temperature_increase_K("C")

    for key in ["heating_setpoint", "heating_setback"]:
        if key in setpoints:
            setpoints[key] = float(setpoints[key]) + h_delta
    for key in ["cooling_setpoint", "cooling_setback"]:
        if key in setpoints:
            setpoints[key] = float(setpoints[key]) - c_delta

    return adjusted


def prepare_emission_loads(
    hourly_sim: pd.DataFrame,
    hourly_sim_emission: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare baseline and equivalent-setpoint loads for EN 15316-2."""

    loads = pd.DataFrame(index=hourly_sim.index)
    loads["T_ext"] = hourly_sim["T_ext"].astype(float)
    loads["T_op"] = hourly_sim["T_op"].astype(float)

    if "Q_H" in hourly_sim:
        loads["Q_H_kWh"] = hourly_sim["Q_H"].astype(float) / 1000.0
    else:
        loads["Q_H_kWh"] = hourly_sim["Q_HC"].clip(lower=0).astype(float) / 1000.0

    if "Q_C" in hourly_sim:
        loads["Q_C_kWh"] = hourly_sim["Q_C"].astype(float) / 1000.0
    else:
        loads["Q_C_kWh"] = (-hourly_sim["Q_HC"].clip(upper=0)).astype(float) / 1000.0

    emission_aligned = hourly_sim_emission.reindex(loads.index)
    if "Q_H" in emission_aligned:
        loads["Q_H_em_out_inc_kWh"] = emission_aligned["Q_H"].astype(float) / 1000.0
    else:
        loads["Q_H_em_out_inc_kWh"] = (
            emission_aligned["Q_HC"].clip(lower=0).astype(float) / 1000.0
        )

    if "Q_C" in emission_aligned:
        loads["Q_C_em_out_inc_kWh"] = emission_aligned["Q_C"].astype(float) / 1000.0
    else:
        loads["Q_C_em_out_inc_kWh"] = (
            -emission_aligned["Q_HC"].clip(upper=0).astype(float) / 1000.0
        )

    return loads


def _distribution_geometry(building: dict) -> tuple[float, float, float, int]:
    """Approximate rectangular Annex B dimensions from the example building."""

    n_floors = max(int(building["building"].get("n_floors", 1)), 1)
    floor_height = float(building["building"].get("height", 3.0)) / n_floors
    footprint_area = max(float(building["building"]["net_floor_area"]) / n_floors, 1.0)
    aspect_ratio = 1.4
    width = float(np.sqrt(footprint_area / aspect_ratio))
    length = footprint_area / width
    return length, width, floor_height, n_floors


def _pipe_psi_by_scenario(scenario: str) -> tuple[float, float, float]:
    return 0.20, 0.30, 0.30


def _space_distribution_sections(
    scenario: str,
    building: dict,
    service: str,
) -> tuple[list[dict], float]:
    """Build Annex B two-pipe sections V, S and A for the example building."""

    length, width, floor_height, n_floors = _distribution_geometry(building)
    psi_v, psi_s, psi_a = _pipe_psi_by_scenario(scenario)
    compact_routing_factor = 0.75
    section_v = compact_routing_factor * (2.0 * length + 0.0325 * length * width + 6.0)
    section_s = compact_routing_factor * (0.025 * length * width * floor_height * n_floors)
    section_a = compact_routing_factor * (0.55 * length * width * n_floors)
    ambient_v = 13.0 if service == "heating" else 26.0
    ambient_zone = 20.0 if service == "heating" else 26.0
    max_length = length + width + floor_height * n_floors + 10.0
    return (
        [
            {
                "name": "V base distributor",
                "length_m": section_v,
                "linear_thermal_transmittance_W_mK": psi_v,
                "ambient_temperature_C": ambient_v,
                "recoverable": False,
            },
            {
                "name": "S vertical shafts",
                "length_m": section_s,
                "linear_thermal_transmittance_W_mK": psi_s,
                "ambient_temperature_C": ambient_zone,
                "recoverable": True,
            },
            {
                "name": "A connection pipes",
                "length_m": section_a,
                "linear_thermal_transmittance_W_mK": psi_a,
                "ambient_temperature_C": ambient_zone,
                "recoverable": True,
            },
        ],
        max_length,
    )


def _dhw_distribution_sections(scenario: str, building: dict) -> tuple[list[dict], float]:
    """Build compact DHW distribution sections from EN 15316-3 Annex B."""

    length, width, floor_height, n_floors = _distribution_geometry(building)
    psi_v, psi_s, psi_a = _pipe_psi_by_scenario(scenario)
    circulation_fraction = 0.25
    section_v = circulation_fraction * (2.0 * length + 0.0125 * length * width)
    section_s = circulation_fraction * (0.075 * length * width * n_floors * floor_height)
    section_a = 0.075 * length * width * n_floors
    max_length = length + floor_height * n_floors + 2.5
    return (
        [
            {
                "name": "V compact DHW loop",
                "length_m": section_v,
                "linear_thermal_transmittance_W_mK": psi_v,
                "ambient_temperature_C": 13.0,
                "recoverable": False,
            },
            {
                "name": "S compact DHW shaft",
                "length_m": section_s,
                "linear_thermal_transmittance_W_mK": psi_s,
                "ambient_temperature_C": 20.0,
                "recoverable": True,
            },
            {
                "name": "A DHW branch pipes",
                "length_m": section_a,
                "linear_thermal_transmittance_W_mK": psi_a,
                "ambient_temperature_C": 20.0,
                "recoverable": True,
            },
        ],
        max_length,
    )


def distribution_system_config(scenario: str, building: dict) -> dict:
    """Scenario-specific EN 15316-3 distribution assumptions."""

    heating_sections, heating_max_length = _space_distribution_sections(
        scenario,
        building,
        service="heating",
    )
    cooling_sections, cooling_max_length = _space_distribution_sections(
        scenario,
        building,
        service="cooling",
    )
    dhw_sections, dhw_max_length = _dhw_distribution_sections(scenario, building)

    config = {
        "demand_unit": "kWh",
        "heating": {
            "pipe_sections": heating_sections,
            "supply_temperature_C": 45.0,
            "return_temperature_C": 35.0,
            "design_deltaT_K": 10.0,
            "nominal_power_kW": 10.0,
            "max_length_m": heating_max_length,
            "pressure_loss_per_m_kPa": 0.10,
            "additional_pressure_kPa": 6.0,
            "resistance_ratio": 0.30,
            "pump_control_code": 4,
            "eei": 0.23,
            "recoverable_aux_fraction": 0.25,
        },
        "cooling": {
            "pipe_sections": cooling_sections,
            "supply_temperature_C": 7.0,
            "return_temperature_C": 12.0,
            "design_deltaT_K": 5.0,
            "nominal_power_kW": 8.0,
            "max_length_m": cooling_max_length,
            "pressure_loss_per_m_kPa": 0.10,
            "additional_pressure_kPa": 6.0,
            "resistance_ratio": 0.30,
            "pump_control_code": 3,
            "eei": 0.23,
            "recoverable_aux_fraction": 0.25,
        },
        "dhw": {
            "pipe_sections": dhw_sections,
            "dhw_temperature_C": 55.0,
            "dhw_return_deltaT_K": 5.0,
            "design_flow_m3_h": 0.12,
            "max_length_m": dhw_max_length,
            "pressure_loss_per_m_kPa": 0.10,
            "additional_pressure_kPa": 6.0,
            "resistance_ratio": 0.30,
            "pump_control_code": 3,
            "pump_label_power_kW": 0.020,
            "part_load_mode": "constant_when_on",
            "recoverable_aux_fraction": 0.25,
        },
    }

    if scenario == "bolzano":
        config["heating"]["nominal_power_kW"] = 11.0
        config["cooling"]["nominal_power_kW"] = 6.5
        config["dhw"]["dhw_temperature_C"] = 55.0

    return config


def storage_system_config(scenario: str) -> dict:
    """Scenario-specific EN 15316-5 storage assumptions.

    The example keeps EN 15316-5 connection losses at 1.0 because pipe and valve
    losses are already represented explicitly in the EN 15316-3 distribution
    module. Declared daily standby losses are converted internally to H_sto_ls.
    """

    config = {
        "demand_unit": "kWh",
        "heating": {
            "storage_volume_l": 80.0,
            "set_temperature_C": 45.0,
            "output_temperature_C": 45.0,
            "ambient_temperature_C": 16.0,
            "standby_loss_kWh_per_day_ref": 0.15,
            "standby_set_temperature_ref_C": 45.0,
            "standby_ambient_temperature_ref_C": 20.0,
            "connection_loss_factor": 1.0,
            "standby_loss_adaptation_factor": 1.0,
            "thermal_loss_room_fraction": 0.75,
            "auxiliary_to_medium_fraction": 0.25,
            "input_pump_power_kW": 0.025,
            "input_pump_flow_m3_h": 0.90,
            "input_pump_deltaT_K": 10.0,
        },
        "dhw": {
            "storage_volume_l": 180.0,
            "set_temperature_C": 55.0,
            "output_temperature_C": 55.0,
            "ambient_temperature_C": 16.0,
            "standby_loss_kWh_per_day_ref": 0.90,
            "standby_set_temperature_ref_C": 55.0,
            "standby_ambient_temperature_ref_C": 20.0,
            "connection_loss_factor": 1.0,
            "standby_loss_adaptation_factor": 1.0,
            "thermal_loss_room_fraction": 0.75,
            "auxiliary_to_medium_fraction": 0.25,
            "input_pump_power_kW": 0.025,
            "input_pump_flow_m3_h": 0.50,
            "input_pump_deltaT_K": 5.0,
        },
    }

    if scenario == "bolzano":
        config["heating"].update(
            {
                "storage_volume_l": 100.0,
                "standby_loss_kWh_per_day_ref": 0.20,
                "input_pump_power_kW": 0.030,
                "input_pump_flow_m3_h": 1.00,
            }
        )
        config["dhw"]["standby_loss_kWh_per_day_ref"] = 0.95

    return config


def prepare_heat_pump_loads(hourly_sim: pd.DataFrame, dhw_kWh: pd.Series) -> pd.DataFrame:
    """Convert ISO52016 Wh outputs to the heat-pump module kWh inputs."""

    loads = pd.DataFrame(index=hourly_sim.index)
    loads["T_ext"] = hourly_sim["T_ext"].astype(float)
    loads["T_op"] = hourly_sim["T_op"].astype(float)
    if "Q_H" in hourly_sim:
        loads["Q_H_kWh"] = hourly_sim["Q_H"].astype(float) / 1000.0
    else:
        loads["Q_H_kWh"] = hourly_sim["Q_HC"].clip(lower=0).astype(float) / 1000.0

    if "Q_C" in hourly_sim:
        loads["Q_C_kWh"] = hourly_sim["Q_C"].astype(float) / 1000.0
    else:
        loads["Q_C_kWh"] = (-hourly_sim["Q_HC"].clip(upper=0)).astype(float) / 1000.0

    loads["Q_W_kWh"] = dhw_kWh.reindex(loads.index).astype(float)
    return loads


def ensure_heating_and_cooling(loads: pd.DataFrame) -> None:
    heating = float(loads["Q_H_kWh"].sum())
    cooling = float(loads["Q_C_kWh"].sum())
    dhw = float(loads["Q_W_kWh"].sum())
    if heating <= 0 or cooling <= 0:
        raise RuntimeError(
            "The ISO52016 run did not produce both heating and cooling demand. "
            f"Calculated heating={heating:.2f} kWh, cooling={cooling:.2f} kWh. "
            "Use a warmer/cooler EPW file, lower the cooling setpoint, or adapt "
            "the example building gains/glazing."
        )
    if dhw <= 0:
        raise RuntimeError("The DHW calculation returned zero demand; check the DHW inputs.")


def allocate_bin_outputs_to_hours(loads: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """Map bin-method results back to hours for visual inspection.

    EN 15316-4-2 is evaluated by temperature bins. This helper does not replace
    the bin calculation; it distributes each bin total back to its member hours
    in proportion to hourly service demand, so users can inspect when the annual
    electricity and backup energy are effectively caused.
    """

    if bins.empty:
        return loads.copy()

    bin_width = float((bins["bin_upper_C"] - bins["bin_lower_C"]).median())
    out = loads.copy()
    out["bin_lower_C"] = np.floor(out["T_ext"].astype(float) / bin_width) * bin_width

    service_specs = {
        "H": "Q_H_kWh",
        "W": "Q_W_kWh",
        "C": "Q_C_kWh",
    }
    service_result_cols = [
        "Q_hp_out_kWh",
        "Q_backup_out_kWh",
        "Q_unmet_kWh",
        "E_hp_in_kWh",
        "E_backup_in_kWh",
        "Q_environment_in_kWh",
        "Q_rejected_kWh",
    ]

    for prefix in service_specs:
        for col in service_result_cols:
            out[f"{prefix}_{col}"] = 0.0
        out[f"{prefix}_performance"] = np.nan
        out[f"{prefix}_capacity_kW"] = np.nan

    for col in ["W_HW_gen_aux_kWh", "W_C_gen_aux_kWh"]:
        out[col] = 0.0

    for _, bin_row in bins.iterrows():
        mask = out["bin_lower_C"].eq(float(bin_row["bin_lower_C"]))
        if not bool(mask.any()):
            continue

        n_hours = int(mask.sum())
        hour_weight = pd.Series(1.0 / n_hours, index=out.index[mask])

        for prefix, load_col in service_specs.items():
            loads_in_bin = out.loc[mask, load_col].clip(lower=0.0)
            total_load = float(loads_in_bin.sum())
            if total_load > 0:
                weights = loads_in_bin / total_load
            else:
                weights = hour_weight

            for col in service_result_cols:
                bin_value = float(bin_row.get(f"{prefix}_{col}", 0.0) or 0.0)
                out.loc[mask, f"{prefix}_{col}"] = weights * bin_value

            out.loc[mask, f"{prefix}_performance"] = bin_row.get(f"{prefix}_performance")
            out.loc[mask, f"{prefix}_capacity_kW"] = bin_row.get(f"{prefix}_capacity_kW")

        out.loc[mask, "W_HW_gen_aux_kWh"] = hour_weight * float(
            bin_row.get("W_HW_gen_aux_kWh", 0.0) or 0.0
        )
        out.loc[mask, "W_C_gen_aux_kWh"] = hour_weight * float(
            bin_row.get("W_C_gen_aux_kWh", 0.0) or 0.0
        )

    out["E_HW_total_kWh"] = (
        out["H_E_hp_in_kWh"]
        + out["W_E_hp_in_kWh"]
        + out["H_E_backup_in_kWh"]
        + out["W_E_backup_in_kWh"]
        + out["W_HW_gen_aux_kWh"]
    )
    out["E_C_total_kWh"] = out["C_E_hp_in_kWh"] + out["C_E_backup_in_kWh"] + out["W_C_gen_aux_kWh"]
    out["E_total_kWh"] = out["E_HW_total_kWh"] + out["E_C_total_kWh"]
    return out


def _write_plot(fig: go.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=80, b=55),
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path


def create_iso52016_visuals(hourly_sim: pd.DataFrame, output_dir: Path, building_area: float) -> Path | None:
    """Create the existing ISO52016 report page in the visual output folder."""

    iso_dir = output_dir / "visuals" / "iso52016"
    iso_dir.mkdir(parents=True, exist_ok=True)
    try:
        pybui.Graphs_and_report(
            df=hourly_sim,
            season="heating_cooling",
            building_area=building_area,
        ).bui_analysis_page(folder_directory=str(iso_dir), name_file="iso52016_building_report")
    except Exception as exc:
        print(f"ISO52016 visual report could not be generated: {exc}")
        return None
    return iso_dir / "iso52016_building_report.html"


def plot_input_timeseries(
    loads: pd.DataFrame,
    output_dir: Path,
    title: str = "Space Emission Loads and DHW Inputs Sent to the Heat Pump",
) -> Path:
    daily_energy = loads[["Q_H_kWh", "Q_C_kWh", "Q_W_kWh"]].resample("D").sum()
    daily_temp = loads[["T_ext", "T_op"]].resample("D").mean()
    cumulative = daily_energy.cumsum()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily useful thermal demand",
            "Daily mean temperatures",
            "Cumulative useful demand",
        ],
    )
    for col, name, color in [
        ("Q_H_kWh", "Space heating", "#b23b3b"),
        ("Q_C_kWh", "Space cooling", "#2f78b7"),
        ("Q_W_kWh", "DHW", "#e09f3e"),
    ]:
        fig.add_trace(go.Bar(x=daily_energy.index, y=daily_energy[col], name=name, marker_color=color), row=1, col=1)
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[col], mode="lines", name=f"Cumulative {name}", line=dict(color=color)), row=3, col=1)

    fig.add_trace(go.Scatter(x=daily_temp.index, y=daily_temp["T_ext"], mode="lines", name="Outdoor temperature", line=dict(color="#44546a")), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily_temp.index, y=daily_temp["T_op"], mode="lines", name="Operative temperature", line=dict(color="#70ad47")), row=2, col=1)
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="degC", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_layout(title=title)
    return _write_plot(fig, output_dir / "visuals" / "01_inputs_timeseries.html")


def plot_emission_timeseries(
    emission: pybui.EmissionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = emission.timeseries
    daily = hourly[
        [
            "Q_H_em_out_kWh",
            "Q_H_em_ls_kWh",
            "Q_H_em_in_kWh",
            "Q_C_em_out_kWh",
            "Q_C_em_ls_kWh",
            "Q_C_em_in_kWh",
            "W_H_em_aux_kWh",
            "W_C_em_aux_kWh",
        ]
    ].resample("D").sum()
    daily_temp = hourly[
        [
            "T_H_int_ini_C",
            "theta_H_int_inc_C",
            "T_C_int_ini_C",
            "theta_C_int_inc_C",
        ]
    ].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily building needs and emission input loads",
            "Daily emission losses and auxiliary electricity",
            "Baseline and equivalent internal temperatures",
        ],
    )
    for col, name, color in [
        ("Q_H_em_out_kWh", "Heating building need", "#b23b3b"),
        ("Q_H_em_in_kWh", "Heating emission input", "#7f1d1d"),
        ("Q_C_em_out_kWh", "Cooling building need", "#2f78b7"),
        ("Q_C_em_in_kWh", "Cooling emission input", "#1f4e79"),
    ]:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=name, line=dict(color=color)), row=1, col=1)

    for col, name, color in [
        ("Q_H_em_ls_kWh", "Heating emission losses", "#d65f5f"),
        ("Q_C_em_ls_kWh", "Cooling emission losses", "#6fa8dc"),
        ("W_H_em_aux_kWh", "Heating emission auxiliaries", "#808080"),
        ("W_C_em_aux_kWh", "Cooling emission auxiliaries", "#5b9bd5"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color, dash in [
        ("T_H_int_ini_C", "Heating base internal temperature", "#b23b3b", "dot"),
        ("theta_H_int_inc_C", "Heating equivalent internal temperature", "#7f1d1d", "solid"),
        ("T_C_int_ini_C", "Cooling base internal temperature", "#2f78b7", "dot"),
        ("theta_C_int_inc_C", "Cooling equivalent internal temperature", "#1f4e79", "solid"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=daily_temp.index,
                y=daily_temp[col],
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash),
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=3, col=1)
    fig.update_layout(title="EN 15316-2 Emission System Time Series")
    return _write_plot(fig, output_dir / "visuals" / "02_emission_15316_2_timeseries.html")


def plot_emission_monthly(
    emission: pybui.EmissionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = emission.timeseries
    monthly = hourly[
        [
            "Q_H_em_out_kWh",
            "Q_H_em_temp_effect_kWh",
            "Q_H_emb_ls_kWh",
            "Q_H_em_in_kWh",
            "Q_C_em_out_kWh",
            "Q_C_em_temp_effect_kWh",
            "Q_C_emb_ls_kWh",
            "Q_C_em_in_kWh",
            "W_H_em_aux_kWh",
            "W_C_em_aux_kWh",
        ]
    ].resample("ME").sum()
    monthly["e_H_em_ls"] = monthly["Q_H_em_in_kWh"] / monthly["Q_H_em_out_kWh"].replace(0, np.nan)
    monthly["e_C_em_ls"] = monthly["Q_C_em_in_kWh"] / monthly["Q_C_em_out_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "Monthly EN 15316-2 emission balance",
            "Monthly emission expenditure factors",
        ],
    )
    for col, name, color in [
        ("Q_H_em_out_kWh", "Heating building need", "#b23b3b"),
        ("Q_H_em_temp_effect_kWh", "Heating control/emitter effect", "#d65f5f"),
        ("Q_H_emb_ls_kWh", "Heating embedded loss", "#a64242"),
        ("Q_C_em_out_kWh", "Cooling building need", "#2f78b7"),
        ("Q_C_em_temp_effect_kWh", "Cooling control/emitter effect", "#6fa8dc"),
        ("Q_C_emb_ls_kWh", "Cooling embedded loss", "#1f4e79"),
        ("W_H_em_aux_kWh", "Heating auxiliaries", "#808080"),
        ("W_C_em_aux_kWh", "Cooling auxiliaries", "#5b9bd5"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["e_H_em_ls"],
            mode="lines+markers",
            name="Heating expenditure factor",
            line=dict(color="#7f1d1d"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["e_C_em_ls"],
            mode="lines+markers",
            name="Cooling expenditure factor",
            line=dict(color="#2f78b7"),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly EN 15316-2 Emission Summary", barmode="relative")
    return _write_plot(fig, output_dir / "visuals" / "03_emission_15316_2_monthly.html")


def plot_distribution_timeseries(
    distribution: pybui.DistributionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = distribution.timeseries
    daily = hourly[
        [
            "Q_H_dis_out_kWh",
            "Q_H_dis_ls_kWh",
            "Q_H_dis_in_kWh",
            "Q_C_dis_out_kWh",
            "Q_C_dis_ls_kWh",
            "Q_C_dis_in_kWh",
            "Q_W_dis_out_kWh",
            "Q_W_dis_ls_kWh",
            "Q_W_dis_in_kWh",
            "W_H_dis_aux_kWh",
            "W_C_dis_aux_kWh",
            "W_W_dis_aux_kWh",
        ]
    ].resample("D").sum()
    temps = hourly[
        ["theta_H_dis_mean_C", "theta_C_dis_mean_C", "theta_W_dis_mean_C"]
    ].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily distribution output and generator-side input loads",
            "Daily distribution thermal losses and pump electricity",
            "Daily mean distribution water temperatures",
        ],
    )
    for col, name, color in [
        ("Q_H_dis_out_kWh", "Heating downstream output", "#b23b3b"),
        ("Q_H_dis_in_kWh", "Heating generator-side input", "#7f1d1d"),
        ("Q_C_dis_out_kWh", "Cooling downstream output", "#2f78b7"),
        ("Q_C_dis_in_kWh", "Cooling generator-side input", "#1f4e79"),
        ("Q_W_dis_out_kWh", "DHW downstream output", "#e09f3e"),
        ("Q_W_dis_in_kWh", "DHW generator-side input", "#9c6500"),
    ]:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=name, line=dict(color=color)), row=1, col=1)

    for col, name, color in [
        ("Q_H_dis_ls_kWh", "Heating pipe losses", "#d65f5f"),
        ("Q_C_dis_ls_kWh", "Cooling pipe gains", "#6fa8dc"),
        ("Q_W_dis_ls_kWh", "DHW pipe losses", "#f2bf6d"),
        ("W_H_dis_aux_kWh", "Heating pump electricity", "#808080"),
        ("W_C_dis_aux_kWh", "Cooling pump electricity", "#5b9bd5"),
        ("W_W_dis_aux_kWh", "DHW pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color in [
        ("theta_H_dis_mean_C", "Heating water", "#b23b3b"),
        ("theta_C_dis_mean_C", "Cooling water", "#2f78b7"),
        ("theta_W_dis_mean_C", "DHW water", "#e09f3e"),
    ]:
        fig.add_trace(go.Scatter(x=temps.index, y=temps[col], mode="lines", name=name, line=dict(color=color)), row=3, col=1)

    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=3, col=1)
    fig.update_layout(title="EN 15316-3 Distribution System Time Series")
    return _write_plot(fig, output_dir / "visuals" / "04_distribution_15316_3_timeseries.html")


def plot_distribution_monthly(
    distribution: pybui.DistributionSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = distribution.timeseries
    monthly = hourly[
        [
            "Q_H_dis_out_kWh",
            "Q_H_dis_ls_kWh",
            "Q_H_dis_in_kWh",
            "Q_C_dis_out_kWh",
            "Q_C_dis_ls_kWh",
            "Q_C_dis_in_kWh",
            "Q_W_dis_out_kWh",
            "Q_W_dis_ls_kWh",
            "Q_W_dis_in_kWh",
            "W_H_dis_aux_kWh",
            "W_C_dis_aux_kWh",
            "W_W_dis_aux_kWh",
        ]
    ].resample("ME").sum()
    monthly["e_H_dis"] = monthly["Q_H_dis_in_kWh"] / monthly["Q_H_dis_out_kWh"].replace(0, np.nan)
    monthly["e_C_dis"] = monthly["Q_C_dis_in_kWh"] / monthly["Q_C_dis_out_kWh"].replace(0, np.nan)
    monthly["e_W_dis"] = monthly["Q_W_dis_in_kWh"] / monthly["Q_W_dis_out_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "Monthly EN 15316-3 distribution balance",
            "Monthly distribution expenditure factors",
        ],
    )
    for col, name, color in [
        ("Q_H_dis_out_kWh", "Heating downstream output", "#b23b3b"),
        ("Q_H_dis_ls_kWh", "Heating pipe losses", "#d65f5f"),
        ("Q_C_dis_out_kWh", "Cooling downstream output", "#2f78b7"),
        ("Q_C_dis_ls_kWh", "Cooling pipe gains", "#6fa8dc"),
        ("Q_W_dis_out_kWh", "DHW downstream output", "#e09f3e"),
        ("Q_W_dis_ls_kWh", "DHW pipe losses", "#f2bf6d"),
        ("W_H_dis_aux_kWh", "Heating pump electricity", "#808080"),
        ("W_C_dis_aux_kWh", "Cooling pump electricity", "#5b9bd5"),
        ("W_W_dis_aux_kWh", "DHW pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("e_H_dis", "Heating distribution factor", "#7f1d1d"),
        ("e_C_dis", "Cooling distribution factor", "#2f78b7"),
        ("e_W_dis", "DHW distribution factor", "#9c6500"),
    ]:
        fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly[col], mode="lines+markers", name=name, line=dict(color=color)),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly EN 15316-3 Distribution Summary", barmode="relative")
    return _write_plot(fig, output_dir / "visuals" / "05_distribution_15316_3_monthly.html")


def plot_storage_timeseries(
    storage: pybui.StorageSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = storage.timeseries
    daily = hourly[
        [
            "Q_H_sto_out_kWh",
            "Q_H_sto_ls_kWh",
            "Q_H_sto_in_kWh",
            "Q_W_sto_out_kWh",
            "Q_W_sto_ls_kWh",
            "Q_W_sto_in_kWh",
            "W_H_sto_aux_kWh",
            "W_W_sto_aux_kWh",
        ]
    ].resample("D").sum()
    temps = hourly[
        ["theta_H_sto_set_C", "theta_W_sto_set_C", "theta_H_sto_amb_C", "theta_W_sto_amb_C"]
    ].resample("D").mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily storage output and generator-side input loads",
            "Daily storage standing losses and pump electricity",
            "Daily storage setpoint and ambient temperatures",
        ],
    )
    for col, name, color in [
        ("Q_H_sto_out_kWh", "Heating storage output", "#b23b3b"),
        ("Q_H_sto_in_kWh", "Heating generator-side input", "#7f1d1d"),
        ("Q_W_sto_out_kWh", "DHW storage output", "#e09f3e"),
        ("Q_W_sto_in_kWh", "DHW generator-side input", "#9c6500"),
    ]:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=name, line=dict(color=color)), row=1, col=1)

    for col, name, color in [
        ("Q_H_sto_ls_kWh", "Heating tank losses", "#d65f5f"),
        ("Q_W_sto_ls_kWh", "DHW tank losses", "#f2bf6d"),
        ("W_H_sto_aux_kWh", "Heating storage pump electricity", "#808080"),
        ("W_W_sto_aux_kWh", "DHW storage pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color in [
        ("theta_H_sto_set_C", "Heating storage setpoint", "#b23b3b"),
        ("theta_W_sto_set_C", "DHW storage setpoint", "#e09f3e"),
        ("theta_H_sto_amb_C", "Heating storage ambient", "#7f7f7f"),
        ("theta_W_sto_amb_C", "DHW storage ambient", "#a6a6a6"),
    ]:
        fig.add_trace(go.Scatter(x=temps.index, y=temps[col], mode="lines", name=name, line=dict(color=color)), row=3, col=1)

    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="kWh/day", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=3, col=1)
    fig.update_layout(title="EN 15316-5 Storage System Time Series")
    return _write_plot(fig, output_dir / "visuals" / "06_storage_15316_5_timeseries.html")


def plot_storage_monthly(
    storage: pybui.StorageSimulationResult,
    output_dir: Path,
) -> Path:
    hourly = storage.timeseries
    monthly = hourly[
        [
            "Q_H_sto_out_kWh",
            "Q_H_sto_ls_kWh",
            "Q_H_sto_ls_rbl_kWh",
            "Q_H_sto_ls_nrbl_kWh",
            "Q_H_sto_in_kWh",
            "Q_W_sto_out_kWh",
            "Q_W_sto_ls_kWh",
            "Q_W_sto_ls_rbl_kWh",
            "Q_W_sto_ls_nrbl_kWh",
            "Q_W_sto_in_kWh",
            "W_H_sto_aux_kWh",
            "W_W_sto_aux_kWh",
        ]
    ].resample("ME").sum()
    monthly["e_H_sto"] = monthly["Q_H_sto_in_kWh"] / monthly["Q_H_sto_out_kWh"].replace(0, np.nan)
    monthly["e_W_sto"] = monthly["Q_W_sto_in_kWh"] / monthly["Q_W_sto_out_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "Monthly EN 15316-5 storage balance",
            "Monthly storage expenditure factors",
        ],
    )
    for col, name, color in [
        ("Q_H_sto_out_kWh", "Heating storage output", "#b23b3b"),
        ("Q_H_sto_ls_rbl_kWh", "Heating recoverable loss", "#d65f5f"),
        ("Q_H_sto_ls_nrbl_kWh", "Heating non-recoverable loss", "#7f1d1d"),
        ("Q_W_sto_out_kWh", "DHW storage output", "#e09f3e"),
        ("Q_W_sto_ls_rbl_kWh", "DHW recoverable loss", "#f2bf6d"),
        ("Q_W_sto_ls_nrbl_kWh", "DHW non-recoverable loss", "#9c6500"),
        ("W_H_sto_aux_kWh", "Heating storage pump electricity", "#808080"),
        ("W_W_sto_aux_kWh", "DHW storage pump electricity", "#b7b7b7"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("e_H_sto", "Heating storage factor", "#7f1d1d"),
        ("e_W_sto", "DHW storage factor", "#9c6500"),
    ]:
        fig.add_trace(
            go.Scatter(x=monthly.index, y=monthly[col], mode="lines+markers", name=name, line=dict(color=color)),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly EN 15316-5 Storage Summary", barmode="relative")
    return _write_plot(fig, output_dir / "visuals" / "07_storage_15316_5_monthly.html")


def plot_heat_pump_hourly(allocated: pd.DataFrame, output_dir: Path) -> Path:
    daily = allocated[
        [
            "E_HW_total_kWh",
            "E_C_total_kWh",
            "H_E_hp_in_kWh",
            "W_E_hp_in_kWh",
            "C_E_hp_in_kWh",
            "H_E_backup_in_kWh",
            "W_E_backup_in_kWh",
            "W_HW_gen_aux_kWh",
            "W_C_gen_aux_kWh",
        ]
    ].resample("D").sum()

    perf = allocated[["H_performance", "W_performance", "C_performance"]].resample("D").mean()
    cumulative = daily[["E_HW_total_kWh", "E_C_total_kWh"]].cumsum()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Daily heat-pump electricity by service",
            "Daily mean bin COP/EER assigned to hours",
            "Cumulative final electricity",
        ],
    )
    bars = [
        ("H_E_hp_in_kWh", "Heating compressor", "#b23b3b"),
        ("W_E_hp_in_kWh", "DHW compressor", "#e09f3e"),
        ("C_E_hp_in_kWh", "Cooling compressor", "#2f78b7"),
        ("H_E_backup_in_kWh", "Heating backup", "#7f1d1d"),
        ("W_E_backup_in_kWh", "DHW backup", "#9c6500"),
        ("W_HW_gen_aux_kWh", "Heating+DHW auxiliaries", "#808080"),
        ("W_C_gen_aux_kWh", "Cooling auxiliaries", "#5b9bd5"),
    ]
    for col, name, color in bars:
        fig.add_trace(go.Bar(x=daily.index, y=daily[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("H_performance", "Heating COP", "#b23b3b"),
        ("W_performance", "DHW COP", "#e09f3e"),
        ("C_performance", "Cooling EER", "#2f78b7"),
    ]:
        fig.add_trace(go.Scatter(x=perf.index, y=perf[col], mode="lines", name=name, line=dict(color=color)), row=2, col=1)

    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative["E_HW_total_kWh"], mode="lines", name="Cumulative heating+DHW electricity", line=dict(color="#7f1d1d")), row=3, col=1)
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative["E_C_total_kWh"], mode="lines", name="Cumulative cooling electricity", line=dict(color="#2f78b7")), row=3, col=1)
    fig.update_yaxes(title_text="kWh/day", row=1, col=1)
    fig.update_yaxes(title_text="COP / EER", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_layout(title="Allocated Hourly View of Heat-Pump Outputs")
    return _write_plot(fig, output_dir / "visuals" / "02_heat_pump_timeseries.html")


def plot_monthly_summary(loads: pd.DataFrame, allocated: pd.DataFrame, output_dir: Path) -> Path:
    monthly_loads = loads[["Q_H_kWh", "Q_C_kWh", "Q_W_kWh"]].resample("ME").sum()
    monthly_energy = allocated[["E_HW_total_kWh", "E_C_total_kWh"]].resample("ME").sum()
    monthly = pd.concat([monthly_loads, monthly_energy], axis=1)
    monthly["SPF_HW_month"] = (monthly["Q_H_kWh"] + monthly["Q_W_kWh"]) / monthly["E_HW_total_kWh"].replace(0, np.nan)
    monthly["SEER_C_month"] = monthly["Q_C_kWh"] / monthly["E_C_total_kWh"].replace(0, np.nan)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["Monthly useful demand and final electricity", "Monthly SPF / SEER"],
    )
    for col, name, color in [
        ("Q_H_kWh", "Heating demand", "#b23b3b"),
        ("Q_W_kWh", "DHW demand", "#e09f3e"),
        ("Q_C_kWh", "Cooling demand", "#2f78b7"),
        ("E_HW_total_kWh", "Heating+DHW electricity", "#595959"),
        ("E_C_total_kWh", "Cooling electricity", "#5b9bd5"),
    ]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[col], name=name, marker_color=color), row=1, col=1)

    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["SPF_HW_month"], mode="lines+markers", name="SPF heating+DHW", line=dict(color="#7f1d1d")), row=2, col=1)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["SEER_C_month"], mode="lines+markers", name="SEER cooling", line=dict(color="#2f78b7")), row=2, col=1)
    fig.update_yaxes(title_text="kWh/month", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(title="Monthly Aggregates")
    return _write_plot(fig, output_dir / "visuals" / "03_monthly_summary.html")


def plot_bin_balance(bins: pd.DataFrame, output_dir: Path) -> Path:
    x = bins["bin_center_C"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Thermal demand/output by outdoor-temperature bin",
            "Input electricity by bin",
            "Auxiliaries, losses and unmet demand",
        ],
    )
    for col, name, color in [
        ("H_Q_gen_out_kWh", "Heating demand", "#b23b3b"),
        ("W_Q_gen_out_kWh", "DHW demand", "#e09f3e"),
        ("C_Q_gen_out_kWh", "Cooling demand", "#2f78b7"),
        ("H_Q_hp_out_kWh", "Heating from HP", "#d65f5f"),
        ("W_Q_hp_out_kWh", "DHW from HP", "#f2bf6d"),
        ("C_Q_hp_out_kWh", "Cooling from HP", "#6fa8dc"),
    ]:
        if col in bins:
            fig.add_trace(go.Bar(x=x, y=bins[col], name=name, marker_color=color), row=1, col=1)

    for col, name, color in [
        ("H_E_hp_in_kWh", "Heating compressor", "#b23b3b"),
        ("W_E_hp_in_kWh", "DHW compressor", "#e09f3e"),
        ("C_E_hp_in_kWh", "Cooling compressor", "#2f78b7"),
        ("H_E_backup_in_kWh", "Heating backup", "#7f1d1d"),
        ("W_E_backup_in_kWh", "DHW backup", "#9c6500"),
    ]:
        if col in bins:
            fig.add_trace(go.Bar(x=x, y=bins[col], name=name, marker_color=color), row=2, col=1)

    for col, name, color in [
        ("W_HW_gen_aux_kWh", "Heating+DHW auxiliaries", "#808080"),
        ("W_C_gen_aux_kWh", "Cooling auxiliaries", "#5b9bd5"),
        ("Q_HW_gen_ls_tot_kWh", "Heating+DHW gen losses", "#a6a6a6"),
        ("H_Q_unmet_kWh", "Heating unmet", "#ff0000"),
        ("W_Q_unmet_kWh", "DHW unmet", "#ff9900"),
        ("C_Q_unmet_kWh", "Cooling unmet", "#00a2ff"),
    ]:
        if col in bins:
            fig.add_trace(go.Bar(x=x, y=bins[col], name=name, marker_color=color), row=3, col=1)

    fig.update_xaxes(title_text="Outdoor-temperature bin center [degC]", row=3, col=1)
    fig.update_yaxes(title_text="kWh", row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_layout(title="Heat-Pump Bin Energy Balance", barmode="group")
    return _write_plot(fig, output_dir / "visuals" / "04_bin_energy_balance.html")


def plot_bin_performance(bins: pd.DataFrame, output_dir: Path) -> Path:
    x = bins["bin_center_C"]
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            "Direct heating and DHW COP by bin",
            "Direct cooling EER by bin",
            "Capacity by bin",
            "Runtime and effective hours",
        ],
    )
    for prefix, label, color in [
        ("H", "Heating", "#b23b3b"),
        ("W", "DHW", "#e09f3e"),
        ("C", "Cooling", "#2f78b7"),
    ]:
        performance_row = 2 if prefix == "C" else 1
        performance_name = f"{label} EER" if prefix == "C" else f"{label} COP"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bins[f"{prefix}_performance"],
                mode="lines+markers",
                name=performance_name,
                line=dict(color=color),
            ),
            row=performance_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bins[f"{prefix}_capacity_kW"],
                mode="lines+markers",
                name=f"{label} capacity",
                line=dict(color=color),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=bins[f"{prefix}_hp_runtime_h"],
                name=f"{label} HP runtime",
                marker_color=color,
            ),
            row=4,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=bins["effective_hours"],
            mode="lines",
            name="Effective bin hours",
            line=dict(color="#404040", dash="dash"),
        ),
        row=4,
        col=1,
    )
    fig.update_xaxes(title_text="Outdoor-temperature bin center [degC]", row=4, col=1)
    fig.update_yaxes(title_text="COP [-]", row=1, col=1)
    fig.update_yaxes(title_text="EER [-]", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    fig.update_yaxes(title_text="hours", row=4, col=1)
    fig.update_layout(
        title="Heat-Pump Product Map and Runtime Inspection",
        height=980,
    )
    return _write_plot(fig, output_dir / "visuals" / "05_bin_performance.html")


def plot_summary_sankey(summary: dict[str, float], output_dir: Path) -> Path:
    labels = [
        "Ambient/source heat",
        "HP electricity",
        "Backup electricity",
        "Aux electricity",
        "Heat pump",
        "Backup heater",
        "Useful heating+DHW",
        "Generator/aux losses",
        "Useful cooling extracted",
        "Cooling electricity",
        "Rejected heat",
    ]
    index = {label: i for i, label in enumerate(labels)}
    flows: list[tuple[str, str, float]] = [
        ("Ambient/source heat", "Heat pump", summary.get("QHW_environment_in_kWh", 0.0)),
        ("HP electricity", "Heat pump", summary.get("EH_hp_in_kWh", 0.0) + summary.get("EW_hp_in_kWh", 0.0)),
        ("Backup electricity", "Backup heater", summary.get("EHW_backup_in_kWh", 0.0)),
        ("Heat pump", "Useful heating+DHW", summary.get("QHW_hp_out_kWh", 0.0)),
        ("Backup heater", "Useful heating+DHW", summary.get("QHW_backup_out_kWh", 0.0)),
        ("Aux electricity", "Generator/aux losses", summary.get("WHW_gen_aux_kWh", 0.0)),
        ("Useful cooling extracted", "Rejected heat", summary.get("QC_hp_out_kWh", 0.0)),
        ("Cooling electricity", "Rejected heat", summary.get("EC_hp_in_kWh", 0.0) + summary.get("WC_gen_aux_kWh", 0.0)),
    ]
    source = []
    target = []
    value = []
    for src, dst, val in flows:
        if val and np.isfinite(val) and val > 0:
            source.append(index[src])
            target.append(index[dst])
            value.append(float(val))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=16, thickness=18),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(title="Heat-Pump Annual Energy Flow Summary")
    return _write_plot(fig, output_dir / "visuals" / "06_energy_flow_sankey.html")


def create_inspection_index(
    output_dir: Path,
    summary: dict[str, float],
    plot_paths: list[Path],
    iso_report: Path | None,
    emission_summary: dict[str, float] | None = None,
    distribution_summary: dict[str, float] | None = None,
    storage_summary: dict[str, float] | None = None,
) -> Path:
    cards = {
        "Heating demand": summary.get("QH_gen_out_kWh", 0.0),
        "DHW demand": summary.get("QW_gen_out_kWh", 0.0),
        "Cooling demand": summary.get("QC_gen_out_kWh", 0.0),
        "Heating+DHW electricity": summary.get("EHW_gen_in_kWh", 0.0) + summary.get("WHW_gen_aux_kWh", 0.0),
        "Cooling electricity": summary.get("EC_gen_in_kWh", 0.0) + summary.get("WC_gen_aux_kWh", 0.0),
        "SPF heating+DHW": summary.get("SPF_HW_gen", np.nan),
        "SEER cooling": summary.get("SEER_C_gen", np.nan),
    }
    if emission_summary:
        cards.update(
            {
                "EN 15316-2 heating losses": emission_summary.get("QH_em_ls_kWh", 0.0),
                "EN 15316-2 cooling losses": emission_summary.get("QC_em_ls_kWh", 0.0),
                "EN 15316-2 auxiliaries": emission_summary.get("W_em_aux_kWh", 0.0),
            }
        )
    if distribution_summary:
        cards.update(
            {
                "EN 15316-3 pipe losses": distribution_summary.get("Q_dis_ls_kWh", 0.0),
                "EN 15316-3 pump electricity": distribution_summary.get("W_dis_aux_kWh", 0.0),
                "EN 15316-3 DHW losses": distribution_summary.get("QW_dis_ls_kWh", 0.0),
            }
        )
    if storage_summary:
        cards.update(
            {
                "EN 15316-5 storage losses": storage_summary.get("Q_sto_ls_kWh", 0.0),
                "EN 15316-5 storage pump electricity": storage_summary.get("W_sto_aux_kWh", 0.0),
                "EN 15316-5 DHW storage losses": storage_summary.get("QW_sto_ls_kWh", 0.0),
            }
        )
    list_items = []
    if iso_report is not None:
        list_items.append(f'<li><a href="{html.escape(str(iso_report.relative_to(output_dir)))}">ISO52016 existing building report</a></li>')
    for path in plot_paths:
        list_items.append(f'<li><a href="{html.escape(str(path.relative_to(output_dir)))}">{html.escape(path.stem.replace("_", " ").title())}</a></li>')

    card_html = "\n".join(
        f"<div class='card'><span>{html.escape(name)}</span><strong>{value:,.2f}</strong></div>"
        for name, value in cards.items()
        if value is not None and np.isfinite(value)
    )
    page_title = (
        "EN 15316-2, EN 15316-3, EN 15316-5 and EN 15316-4-2 Inspection"
        if storage_summary and distribution_summary and emission_summary
        else "EN 15316-3, EN 15316-5 and EN 15316-4-2 Inspection"
        if storage_summary and distribution_summary
        else "EN 15316-5 and EN 15316-4-2 Inspection"
        if storage_summary
        else "EN 15316-2, EN 15316-3 and EN 15316-4-2 Inspection"
        if distribution_summary
        else "EN 15316-2 and EN 15316-4-2 Inspection"
        if emission_summary
        else "Heat Pump EN 15316-4-2 Inspection"
    )
    page_intro = (
        "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, EN 15316-3 distribution losses and pump auxiliaries, "
        "EN 15316-5 heating/DHW storage losses and auxiliaries, DHW profile, "
        "heat-pump bin method, electricity use, backup energy, losses and "
        "seasonal performance."
        if storage_summary and distribution_summary and emission_summary
        else "Open the plots below to inspect the ISO52016 inputs, EN 15316-3 "
        "distribution losses and pump auxiliaries, EN 15316-5 heating/DHW "
        "storage losses and auxiliaries, DHW profile, heat-pump bin method, "
        "electricity use, backup energy, losses and seasonal performance."
        if storage_summary and distribution_summary
        else "Open the plots below to inspect the ISO52016 and DHW inputs, "
        "EN 15316-5 heating/DHW storage losses and auxiliaries, heat-pump bin "
        "method, electricity use, backup energy, losses and seasonal performance."
        if storage_summary
        else "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, EN 15316-3 distribution losses and pump auxiliaries, "
        "DHW profile, heat-pump bin method, electricity use, backup energy, losses "
        "and seasonal performance."
        if distribution_summary
        else
        "Open the plots below to inspect the ISO52016 inputs, EN 15316-2 "
        "emission effects, DHW profile, heat-pump bin method, electricity use, "
        "backup energy, losses and seasonal performance."
        if emission_summary
        else "Open the plots below to inspect the direct ISO52016 and DHW inputs, "
        "heat-pump bin method, electricity use, backup energy, losses and seasonal "
        "performance."
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(page_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 20px 0 28px; }}
    .card {{ border: 1px solid #d7dde5; border-radius: 8px; padding: 14px 16px; background: #f8fafc; }}
    .card span {{ display: block; color: #5b6777; font-size: 13px; margin-bottom: 8px; }}
    .card strong {{ font-size: 22px; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>{html.escape(page_title)}</h1>
  <p>{html.escape(page_intro)}</p>
  <div class="grid">{card_html}</div>
  <h2>Interactive Outputs</h2>
  <ul>{"".join(list_items)}</ul>
</body>
</html>
"""
    index_path = output_dir / "inspection_index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


def create_visual_outputs(
    hourly_sim: pd.DataFrame,
    loads: pd.DataFrame,
    result: pybui.HeatPumpSimulationResult,
    output_dir: Path,
    building_area: float,
    emission_result: pybui.EmissionSimulationResult | None = None,
    distribution_result: pybui.DistributionSimulationResult | None = None,
    storage_result: pybui.StorageSimulationResult | None = None,
) -> Path:
    iso_report = create_iso52016_visuals(hourly_sim, output_dir, building_area)
    allocated = allocate_bin_outputs_to_hours(loads, result.bins)
    allocated.to_csv(output_dir / "heat_pump_hourly_allocated_results.csv")

    input_title = (
        "Storage-Adjusted Loads Sent to the Heat Pump"
        if storage_result is not None
        else "Distribution-Adjusted Loads and DHW Inputs Sent to the Heat Pump"
        if distribution_result is not None
        else "Space Emission Loads and DHW Inputs Sent to the Heat Pump"
        if emission_result is not None
        else "ISO52016 and DHW Inputs Sent to the Heat Pump"
    )
    plot_paths = [plot_input_timeseries(loads, output_dir, title=input_title)]
    if emission_result is not None:
        plot_paths.extend(
            [
                plot_emission_timeseries(emission_result, output_dir),
                plot_emission_monthly(emission_result, output_dir),
            ]
        )
    if distribution_result is not None:
        plot_paths.extend(
            [
                plot_distribution_timeseries(distribution_result, output_dir),
                plot_distribution_monthly(distribution_result, output_dir),
            ]
        )
    if storage_result is not None:
        plot_paths.extend(
            [
                plot_storage_timeseries(storage_result, output_dir),
                plot_storage_monthly(storage_result, output_dir),
            ]
        )
    plot_paths.extend(
        [
            plot_heat_pump_hourly(allocated, output_dir),
            plot_monthly_summary(loads, allocated, output_dir),
            plot_bin_balance(result.bins, output_dir),
            plot_bin_performance(result.bins, output_dir),
            plot_summary_sankey(result.summary, output_dir),
        ]
    )
    emission_summary = emission_result.summary if emission_result is not None else None
    distribution_summary = (
        distribution_result.summary if distribution_result is not None else None
    )
    storage_summary = storage_result.summary if storage_result is not None else None
    return create_inspection_index(
        output_dir,
        result.summary,
        plot_paths,
        iso_report,
        emission_summary=emission_summary,
        distribution_summary=distribution_summary,
        storage_summary=storage_summary,
    )


def resolve_system_methods(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.calculation_path == "full":
        return "en15316-2", "en15316-3", "en15316-5"
    if args.calculation_path == "emission-distribution":
        return "en15316-2", "en15316-3", "simple"
    if args.calculation_path == "emission-storage":
        return "en15316-2", "simple", "en15316-5"
    if args.calculation_path == "distribution-storage":
        return "simple", "en15316-3", "en15316-5"
    if args.calculation_path == "storage-only":
        return "simple", "simple", "en15316-5"
    if args.calculation_path == "emission-only":
        return "en15316-2", "simple", "simple"
    if args.calculation_path == "simple":
        return "simple", "simple", "simple"

    emission_method = args.emission_method or "en15316-2"
    distribution_method = args.distribution_method
    if distribution_method is None:
        distribution_method = "simple" if emission_method == "simple" else "en15316-3"
    storage_method = args.storage_method
    if storage_method is None:
        storage_method = (
            "simple"
            if emission_method == "simple" and distribution_method == "simple"
            else "en15316-5"
        )
    return emission_method, distribution_method, storage_method


def run_example(args: argparse.Namespace) -> None:
    scenario = args.scenario
    emission_method, distribution_method, storage_method = resolve_system_methods(args)
    building = example_building(scenario)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(scenario, emission_method, distribution_method, storage_method)
    )
    dhw_country = args.dhw_calendar_country or SCENARIO_DHW_COUNTRY[scenario]
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_sim = run_iso52016(building, args.weather_source, args.path_weather_file)
    dhw_kWh = make_dhw_profile(
        pd.DatetimeIndex(hourly_sim.index),
        building_area=float(building["building"]["net_floor_area"]),
        country=dhw_country,
    )
    loads = prepare_heat_pump_loads(hourly_sim, dhw_kWh)

    emission_result = None
    if emission_method == "en15316-2":
        emission_calc = pybui.EmissionSystemCalculator(emission_system_config(scenario))
        emission_building = building_with_emission_setpoints(building, emission_calc)
        hourly_sim_emission = run_iso52016(
            emission_building,
            args.weather_source,
            args.path_weather_file,
        )
        emission_loads = prepare_emission_loads(hourly_sim, hourly_sim_emission)
        emission_result = emission_calc.run_timeseries(emission_loads)
        loads["Q_H_building_kWh"] = emission_result.timeseries["Q_H_em_out_kWh"]
        loads["Q_C_building_kWh"] = emission_result.timeseries["Q_C_em_out_kWh"]
        loads["Q_H_em_loss_kWh"] = emission_result.timeseries["Q_H_em_ls_kWh"]
        loads["Q_C_em_loss_kWh"] = emission_result.timeseries["Q_C_em_ls_kWh"]
        loads["W_H_em_aux_kWh"] = emission_result.timeseries["W_H_em_aux_kWh"]
        loads["W_C_em_aux_kWh"] = emission_result.timeseries["W_C_em_aux_kWh"]
        loads["Q_H_kWh"] = emission_result.timeseries["Q_H_em_in_kWh"]
        loads["Q_C_kWh"] = emission_result.timeseries["Q_C_em_in_kWh"]
    elif emission_method != "simple":
        raise ValueError("--emission-method must be 'en15316-2' or 'simple'.")

    distribution_result = None
    if distribution_method == "en15316-3":
        distribution_calc = pybui.DistributionSystemCalculator(
            distribution_system_config(scenario, building)
        )
        distribution_result = distribution_calc.run_timeseries(loads)
        loads["Q_H_distribution_out_kWh"] = distribution_result.timeseries["Q_H_dis_out_kWh"]
        loads["Q_C_distribution_out_kWh"] = distribution_result.timeseries["Q_C_dis_out_kWh"]
        loads["Q_W_distribution_out_kWh"] = distribution_result.timeseries["Q_W_dis_out_kWh"]
        loads["Q_H_dis_loss_kWh"] = distribution_result.timeseries["Q_H_dis_ls_kWh"]
        loads["Q_C_dis_loss_kWh"] = distribution_result.timeseries["Q_C_dis_ls_kWh"]
        loads["Q_W_dis_loss_kWh"] = distribution_result.timeseries["Q_W_dis_ls_kWh"]
        loads["W_H_dis_aux_kWh"] = distribution_result.timeseries["W_H_dis_aux_kWh"]
        loads["W_C_dis_aux_kWh"] = distribution_result.timeseries["W_C_dis_aux_kWh"]
        loads["W_W_dis_aux_kWh"] = distribution_result.timeseries["W_W_dis_aux_kWh"]
        loads["Q_H_kWh"] = distribution_result.timeseries["Q_H_dis_in_kWh"]
        loads["Q_C_kWh"] = distribution_result.timeseries["Q_C_dis_in_kWh"]
        loads["Q_W_kWh"] = distribution_result.timeseries["Q_W_dis_in_kWh"]
    elif distribution_method != "simple":
        raise ValueError("--distribution-method must be 'en15316-3' or 'simple'.")

    storage_result = None
    if storage_method == "en15316-5":
        storage_calc = pybui.StorageSystemCalculator(storage_system_config(scenario))
        storage_result = storage_calc.run_timeseries(loads)
        loads["Q_H_storage_out_kWh"] = storage_result.timeseries["Q_H_sto_out_kWh"]
        loads["Q_W_storage_out_kWh"] = storage_result.timeseries["Q_W_sto_out_kWh"]
        loads["Q_H_sto_loss_kWh"] = storage_result.timeseries["Q_H_sto_ls_kWh"]
        loads["Q_W_sto_loss_kWh"] = storage_result.timeseries["Q_W_sto_ls_kWh"]
        loads["W_H_sto_aux_kWh"] = storage_result.timeseries["W_H_sto_aux_kWh"]
        loads["W_W_sto_aux_kWh"] = storage_result.timeseries["W_W_sto_aux_kWh"]
        loads["Q_H_kWh"] = storage_result.timeseries["Q_H_sto_in_kWh"]
        loads["Q_W_kWh"] = storage_result.timeseries["Q_W_sto_in_kWh"]
    elif storage_method != "simple":
        raise ValueError("--storage-method must be 'en15316-5' or 'simple'.")

    ensure_heating_and_cooling(loads)

    heating_map, cooling_map = heat_pump_maps(scenario)
    calc = pybui.HeatPumpSystemCalculator(
        heat_pump_config(
            scenario,
            heating_map,
            cooling_map,
            include_internal_storage_losses=storage_method == "simple",
        )
    )
    result = calc.run_timeseries(loads)

    loads.to_csv(output_dir / "iso52016_loads_with_dhw.csv")
    if emission_result is not None:
        emission_result.timeseries.to_csv(output_dir / "emission_15316_2_hourly_results.csv")
        pd.DataFrame([emission_result.summary]).to_csv(
            output_dir / "emission_15316_2_summary.csv",
            index=False,
        )
    if distribution_result is not None:
        distribution_result.timeseries.to_csv(
            output_dir / "distribution_15316_3_hourly_results.csv"
        )
        pd.DataFrame([distribution_result.summary]).to_csv(
            output_dir / "distribution_15316_3_summary.csv",
            index=False,
        )
    if storage_result is not None:
        storage_result.timeseries.to_csv(output_dir / "storage_15316_5_hourly_results.csv")
        pd.DataFrame([storage_result.summary]).to_csv(
            output_dir / "storage_15316_5_summary.csv",
            index=False,
        )
    result.bins.to_csv(output_dir / "heat_pump_bin_results.csv", index=False)
    pd.DataFrame([result.summary]).to_csv(output_dir / "heat_pump_summary.csv", index=False)
    inspection_index = create_visual_outputs(
        hourly_sim=hourly_sim,
        loads=loads,
        result=result,
        output_dir=output_dir,
        building_area=float(building["building"]["net_floor_area"]),
        emission_result=emission_result,
        distribution_result=distribution_result,
        storage_result=storage_result,
    )

    print(f"\nHeat pump example completed for scenario: {scenario}")
    print(f"Emission calculation mode: {emission_method}")
    print(f"Distribution calculation mode: {distribution_method}")
    print(f"Storage calculation mode: {storage_method}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Visual inspection page: {inspection_index.resolve()}")
    if emission_result is not None:
        print(f"ISO52016 space heating need: {emission_result.summary['QH_em_out_kWh']:,.1f} kWh")
        print(f"EN 15316-2 heating emission losses: {emission_result.summary['QH_em_ls_kWh']:,.1f} kWh")
        print(f"Heat-pump space heating input load: {loads['Q_H_kWh'].sum():,.1f} kWh")
        print(f"ISO52016 space cooling need: {emission_result.summary['QC_em_out_kWh']:,.1f} kWh")
        print(f"EN 15316-2 cooling emission losses: {emission_result.summary['QC_em_ls_kWh']:,.1f} kWh")
        print(f"Heat-pump space cooling input load: {loads['Q_C_kWh'].sum():,.1f} kWh")
        print(f"EN 15316-2 emission auxiliaries: {emission_result.summary['W_em_aux_kWh']:,.1f} kWh")
    else:
        load_label = (
            "Heat-pump input load"
            if distribution_result is not None or storage_result is not None
            else "Demand"
        )
        print(f"Space heating {load_label.lower()}: {loads['Q_H_kWh'].sum():,.1f} kWh")
        print(f"Space cooling {load_label.lower()}: {loads['Q_C_kWh'].sum():,.1f} kWh")
    if distribution_result is not None:
        print(f"EN 15316-3 heating distribution losses: {distribution_result.summary['QH_dis_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-3 cooling distribution losses: {distribution_result.summary['QC_dis_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-3 DHW distribution losses: {distribution_result.summary['QW_dis_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-3 distribution pump auxiliaries: {distribution_result.summary['W_dis_aux_kWh']:,.1f} kWh")
    if storage_result is not None:
        print(f"EN 15316-5 heating storage losses: {storage_result.summary['QH_sto_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-5 DHW storage losses: {storage_result.summary['QW_sto_ls_kWh']:,.1f} kWh")
        print(f"EN 15316-5 storage pump auxiliaries: {storage_result.summary['W_sto_aux_kWh']:,.1f} kWh")
        print(f"DHW storage output demand: {storage_result.summary['QW_sto_out_kWh']:,.1f} kWh")
        print(f"Heat-pump DHW input load: {loads['Q_W_kWh'].sum():,.1f} kWh")
    else:
        print(f"DHW demand: {loads['Q_W_kWh'].sum():,.1f} kWh")
    print(f"Heating+DHW electricity incl. backup: {result.summary['EHW_gen_in_kWh']:,.1f} kWh")
    print(f"Heating+DHW auxiliaries: {result.summary['WHW_gen_aux_kWh']:,.1f} kWh")
    print(f"Cooling electricity: {result.summary['EC_gen_in_kWh']:,.1f} kWh")
    print(f"SPF_HW_gen: {result.summary['SPF_HW_gen']:.2f}")
    print(f"SEER_C_gen: {result.summary['SEER_C_gen']:.2f}")


def parse_args(default_scenario: str = "athens") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_DHW_COUNTRY),
        default=default_scenario,
        help=f"Example scenario to run. Default: {default_scenario}.",
    )
    parser.add_argument(
        "--weather-source",
        choices=["pvgis", "epw"],
        default="pvgis",
        help="Weather source used by ISO52016. Default: pvgis.",
    )
    parser.add_argument(
        "--path-weather-file",
        default=None,
        help="EPW path when --weather-source epw is selected.",
    )
    parser.add_argument(
        "--dhw-calendar-country",
        default=None,
        help="Workalendar country name for the DHW profile. Default: scenario-specific.",
    )
    parser.add_argument(
        "--calculation-path",
        choices=[
            "full",
            "emission-distribution",
            "emission-storage",
            "distribution-storage",
            "storage-only",
            "emission-only",
            "simple",
        ],
        default=None,
        help=(
            "Shortcut for the subsystem chain. 'full' applies EN 15316-2, "
            "EN 15316-3 and EN 15316-5 before heat-pump generation; "
            "'emission-distribution' applies EN 15316-2 and EN 15316-3 but "
            "uses the simple storage treatment inside the heat-pump module; "
            "'emission-storage', 'distribution-storage' and 'storage-only' "
            "isolate selected subsystems; 'emission-only' applies EN 15316-2 "
            "only; 'simple' uses the earlier direct ISO52016/DHW loads and "
            "simple heat-pump storage losses. Default: full."
        ),
    )
    parser.add_argument(
        "--emission-method",
        choices=["en15316-2", "simple"],
        default=None,
        help=(
            "Space-emission calculation mode. 'en15316-2' applies emitter/control "
            "effects before heat-pump generation; 'simple' uses the direct ISO52016 "
            "loads as in the earlier example. Default: en15316-2 unless "
            "--calculation-path is set."
        ),
    )
    parser.add_argument(
        "--distribution-method",
        choices=["en15316-3", "simple"],
        default=None,
        help=(
            "Water-based distribution calculation mode. 'en15316-3' applies pipe "
            "losses and pump auxiliaries before heat-pump generation; 'simple' "
            "bypasses distribution. Default: en15316-3 unless --emission-method "
            "simple or --calculation-path is set."
        ),
    )
    parser.add_argument(
        "--storage-method",
        choices=["en15316-5", "simple"],
        default=None,
        help=(
            "Heating/DHW storage calculation mode. 'en15316-5' applies storage "
            "standing losses and storage pump auxiliaries before heat-pump "
            "generation; 'simple' keeps the earlier simplified storage losses "
            "inside the heat-pump module. Default: en15316-5 unless all upstream "
            "methods are simple or --calculation-path is set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder where CSV and visual outputs are written. "
            "Default: examples/outputs/heat_pump_15316_4_2_<scenario>; simplified "
            "paths add descriptive suffixes."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_example(parse_args())
