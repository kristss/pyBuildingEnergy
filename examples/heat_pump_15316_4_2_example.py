"""Run a heat-pump generation example from ISO52016 and DHW demands.

This script demonstrates the complete sequence:

1. Calculate hourly space heating/cooling needs with ISO52016.
2. Calculate a meaningful hourly DHW need with the DHW module.
3. Run the EN 15316-4-2 heat-pump bin calculation.
4. Save the intermediate loads, bin balance and summary outputs.

Default weather uses PVGIS for the selected scenario. If network access is not
available, run with ``--weather-source epw --path-weather-file path/to/weather.epw``.
"""

from __future__ import annotations

import argparse
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


def default_output_dir(scenario: str) -> Path:
    return REPO_ROOT / "examples" / "outputs" / f"heat_pump_15316_4_2_{scenario}"


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


def plot_input_timeseries(loads: pd.DataFrame, output_dir: Path) -> Path:
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
    fig.update_layout(title="ISO52016 and DHW Inputs Sent to the Heat Pump")
    return _write_plot(fig, output_dir / "visuals" / "01_inputs_timeseries.html")


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
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Heat Pump EN 15316-4-2 Inspection</title>
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
  <h1>Heat Pump EN 15316-4-2 Inspection</h1>
  <p>Open the plots below to inspect the ISO52016 inputs, DHW profile, heat-pump bin method, electricity use, backup energy, losses and seasonal performance.</p>
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
) -> Path:
    iso_report = create_iso52016_visuals(hourly_sim, output_dir, building_area)
    allocated = allocate_bin_outputs_to_hours(loads, result.bins)
    allocated.to_csv(output_dir / "heat_pump_hourly_allocated_results.csv")

    plot_paths = [
        plot_input_timeseries(loads, output_dir),
        plot_heat_pump_hourly(allocated, output_dir),
        plot_monthly_summary(loads, allocated, output_dir),
        plot_bin_balance(result.bins, output_dir),
        plot_bin_performance(result.bins, output_dir),
        plot_summary_sankey(result.summary, output_dir),
    ]
    return create_inspection_index(output_dir, result.summary, plot_paths, iso_report)


def run_example(args: argparse.Namespace) -> None:
    scenario = args.scenario
    building = example_building(scenario)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(scenario)
    dhw_country = args.dhw_calendar_country or SCENARIO_DHW_COUNTRY[scenario]
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_sim = run_iso52016(building, args.weather_source, args.path_weather_file)
    dhw_kWh = make_dhw_profile(
        pd.DatetimeIndex(hourly_sim.index),
        building_area=float(building["building"]["net_floor_area"]),
        country=dhw_country,
    )
    loads = prepare_heat_pump_loads(hourly_sim, dhw_kWh)
    ensure_heating_and_cooling(loads)

    heating_map, cooling_map = heat_pump_maps(scenario)
    calc = pybui.HeatPumpSystemCalculator(
        heat_pump_config(scenario, heating_map, cooling_map)
    )
    result = calc.run_timeseries(loads)

    loads.to_csv(output_dir / "iso52016_loads_with_dhw.csv")
    result.bins.to_csv(output_dir / "heat_pump_bin_results.csv", index=False)
    pd.DataFrame([result.summary]).to_csv(output_dir / "heat_pump_summary.csv", index=False)
    inspection_index = create_visual_outputs(
        hourly_sim=hourly_sim,
        loads=loads,
        result=result,
        output_dir=output_dir,
        building_area=float(building["building"]["net_floor_area"]),
    )

    print(f"\nHeat pump example completed for scenario: {scenario}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Visual inspection page: {inspection_index.resolve()}")
    print(f"Space heating demand: {loads['Q_H_kWh'].sum():,.1f} kWh")
    print(f"Space cooling demand: {loads['Q_C_kWh'].sum():,.1f} kWh")
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
        "--output-dir",
        default=None,
        help=(
            "Folder where CSV and visual outputs are written. "
            "Default: examples/outputs/heat_pump_15316_4_2_<scenario>."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_example(parse_args())
