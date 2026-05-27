# pyBuildingEnergy

![pyBuildingEnergy Logo](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/Logo_pyBuild.png)

## Citation


*Please cite us if you use this library*: 
[![DOI](https://zenodo.org/badge/761715706.svg)](https://zenodo.org/doi/10.5281/zenodo.10887919)

## Documentation **(New)**

Check our **new documentation** in GitHub Pages: [pybuildingenergy docs](https://eurac-eebgroup.github.io/pybuildingenergy-docs/).

## Features

The new EPBD recast provides an update on building performance assessment through a methodology that must take into account various aspects such as the thermal characteristics of the building, the use of energy from renewable sources, building automation and control systems, ventilation, cooling, energy recovery, etc.

The methodology should represent the actual operating conditions, allow for the use of measured energy for accuracy and comparability purposes, and be based on hourly or sub-hourly intervals that take into account the variable conditions significantly impacting the operation and performance of the system, as well as internal conditions.

**pyBuildingEnergy** aims to provide an assessment of building performance both in terms of energy and comfort. In this initial release, it is possible to assess the energy performance of the building using ISO 52016-1:2018. Additional modules will be added for a more comprehensive evaluation of performance, assessing ventilation, renewable energies, systems, etc.

The actual calculation methods for the assessment of building performance are the following:

- [x] the (sensible) energy need for heating and cooling, based on hourly or monthly calculations;
- [ ] the latent energy need for (de-)humidification, based on hourly or monthly calculations;
- [x] the internal temperature, based on hourly calculations;
- [x] the sensible heating and cooling load, based on hourly calculations;
- [ ] the moisture and latent heat load for (de-)humidification, based on hourly calculations;
- [ ] the design sensible heating or cooling load and design latent heat load using an hourly calculation interval;
- [ ] the conditions of the supply air to provide the necessary humidification and dehumidification.

The calculation methods can be used for residential or non-residential buildings, or a part of it, referred to as "the building" or the "assessed object".

ISO 52016-1:2018 also contains specifications for the assessment of thermal zones in the building or in the part of a building. The calculations are performed per thermal zone. In the calculations, the thermal zones can be assumed to be thermally coupled or not. ISO 52016-1:2018 is applicable to buildings at the design stage, to new buildings after construction and to existing buildings in the use phase.

-- 

## Weather Data

The tool can use weather data coming from 2 main sources:

- PVGIS API ([link](https://re.jrc.ec.europa.eu/pvg_tools/en/)) - PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM
- `.epw` file from [Ladybug Tools EPWMap](https://www.ladybug.tools/epwmap/)

More details in the example folder.

## Domestic Hot Water - DHW

- [x] Calculation of volume and energy need for domestic hot water according to ISO 12831-3.
- [ ] Assessment of thermal load based on the type of DHW system.

## Space Emission Systems - EN 15316-2 **(New)**

`EmissionSystemCalculator` evaluates space-heating and water-based space-cooling
emission effects before generation. It can be used to account for emitter and
control effects, equivalent internal-temperature changes, embedded emitter
losses, emission auxiliary electricity and annual emission expenditure factors.

For an audit trail between the standard, code and output files, open
[Emission EN 15316-2 Implementation Audit](docs/emission_15316_2_audit.html).

## Heat Pump Generation - EN 15316-4-2 **(New)**

`HeatPumpSystemCalculator` evaluates a reversible heat-pump generator for:

- space heating,
- domestic hot water (DHW),
- and space cooling with a reversible EER map.

The heating and DHW calculation follows the detailed EN 15316-4-2 bin-method structure: outdoor/source temperature bins, product heating capacity and COP maps, source/sink temperature operating points, runtime/capacity checks, auxiliary energy, storage losses, backup energy and SPF outputs. Cooling is reported separately with the same bin/product-map approach using EER values.

For a clause-by-clause audit trail between the standard, the implementation and the output files, open [Heat Pump EN 15316-4-2 Implementation Audit](docs/heat_pump_15316_4_2_audit.html).

```python
import pandas as pd
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
    "T_ext": [-5, 0, 5, 25],
    "Q_H_kWh": [4.0, 3.0, 1.0, 0.0],
    "Q_C_kWh": [0.0, 0.0, 0.0, 2.0],
    "Q_W_kWh": [0.5, 0.5, 0.5, 0.5],
})

calc = pybui.HeatPumpSystemCalculator({
    "heating_performance_map": heating_map,
    "dhw_performance_map": heating_map,
    "cooling_performance_map": cooling_map,
    "source_type": "air",
    "demand_unit": "kWh",
    "dhw_target_temperature_C": 55,
    "dhw_sink_temperature_C": 55,
    "external_auxiliary_power_W": 100,
})

result = calc.run_timeseries(loads)
print(result.summary["SPF_HW_gen"])
print(result.summary["SEER_C_gen"])
```

### Run The Heat Pump Example

The runnable examples are:

```bash
python -m pip install -r requirements.txt
python examples/heat_pump_15316_4_2_example.py --scenario athens
python examples/heat_pump_15316_4_2_example.py --scenario bolzano
```

There is also a Bolzano convenience wrapper:

```bash
python examples/heat_pump_15316_4_2_bolzano_example.py
```

The Athens scenario uses an Athens PVGIS weather location, a Greece DHW calendar and a 26 C cooling setpoint. The Bolzano scenario uses Bolzano coordinates, an Italy DHW calendar, a tighter solar-exposed top-floor envelope and an air-to-water heat-pump map sized for a 120 m2 residential building.

Each scenario runs ISO52016 for the example building, applies EN 15316-2 emission effects by default, calculates an hourly DHW profile, runs the heat-pump generator calculation and writes:

- `examples/outputs/heat_pump_15316_4_2_<scenario>/iso52016_loads_with_dhw.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/emission_15316_2_hourly_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/emission_15316_2_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/heat_pump_hourly_allocated_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/heat_pump_bin_results.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/heat_pump_summary.csv`
- `examples/outputs/heat_pump_15316_4_2_<scenario>/inspection_index.html`

Open `inspection_index.html` in a browser to inspect the visual outputs. The page links to:

- the existing ISO52016 building report generated with `Graphs_and_report`;
- daily input time series for heating, cooling, DHW and temperatures;
- EN 15316-2 emission time series and monthly aggregate plots;
- allocated heat-pump electricity time series;
- monthly demand, electricity, SPF and SEER summaries;
- bin-method energy balance plots;
- bin COP/EER, capacity and runtime plots;
- an annual energy-flow Sankey diagram.

To reproduce the earlier simple calculation without EN 15316-2 emission effects,
use:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --emission-method simple
python examples/heat_pump_15316_4_2_example.py --scenario bolzano --emission-method simple
```

Simple mode writes to `examples/outputs/heat_pump_15316_4_2_<scenario>_simple`
unless `--output-dir` is specified.

By default the script uses PVGIS weather for the selected scenario, so it needs internet access. To run with a local EPW file instead:

```bash
python examples/heat_pump_15316_4_2_example.py --scenario athens --weather-source epw --path-weather-file path/to/weather.epw
```

The script checks that the ISO52016 run produces both heating and cooling demand and that DHW demand is non-zero before running the heat-pump calculation.

## Primary Energy - Heating System **(New)**

The EN 15316 series covers the calculation method for system energy requirements and system efficiencies. This family of standards is an integral part of the EPB set and covers:

## EN 15316 Modular Structure **(New)**

- [x] EN 15316-1: General and expression of energy performance (Modules M3-1, M3-4, M3-9, M8-1, M8-4)
- [x] EN 15316-2: Emission systems (heating and cooling)
- [ ] EN 15316-3: Distribution systems (DHW, heating, cooling)
- [ ] EN 15316-4-X: Heat generation systems:
  - 4-1: Combustion boilers
  - 4-2: Heat pumps
  - 4-3: Solar thermal and photovoltaic systems
  - 4-4: Cogeneration systems
  - 4-5: District heating
  - 4-7: Biomass
- [ ] EN 15316-5: Storage systems

For space heating, applicable standards include EN 15316-1, EN 15316-2-1, EN 15316-2-3 and the appropriate parts of EN 15316-4 depending on the system type, including losses and control aspects.

## Single zone and Multiple Zones **(New)**
# EN ISO 52016 — Multi-zone Calculation and Adjacent Zones

**EN ISO 52016 defines that:**  
The calculation now allows the definition of several **thermal** and **non-thermal** zones adjacent to the considered zone.


**External Adjacent – Unheated Zone**: It is possible to define an **unheated adjacent zone** in contact with the considered thermal zone.  
The length of the separating wall may be **entirely** or **partially** connected to the considered zone.  

The calculation involves:
1. Determining the **internal temperature** of the non-thermal zone.
2. Evaluating the **heat exchange** with the thermal zone.


**External Adjacent – Heated Zone**: In this case, the wall between the two zones is considered **adiabatic** (no heat exchange).
**Adjusted Coefficient**: To account for the **different temperatures** between zones (e.g., thermal and non-thermal), an **adjusted coefficient** is calculated.


### Assumptions and Simplifications
The standard defines various assumptions specified in section *6.5.3 — Assumptions and specific conditions*.  
In general, it aims to **simplify the zoning** approach by reducing the number of zones to a minimum (ISO EN 52016-2:2018).  

It also emphasizes that:

> *A multi-zone calculation with interactions between the zones requires significant and often arbitrary input data (on transmission properties and air flow direction and size).  
> It can also lead to other technical and procedural complications that add uncertainties to the results.  
> A further complication can be the involvement of different heating, cooling and ventilation systems for different zones, which adds to the complexity and arbitrariness of the input and modelling.*  

**Key Remark**: **Therefore, the benefits of calculations with thermally coupled zones can be smaller than the drawbacks.**

---

## EN 16798-7 & 16798-1 - Natual ventilation and profiles **(New)**

Compute the ventilation heat transfer coefficient [W·K⁻¹] of the thermal zone either: 

- from natural ventilation (ISO 16798-7:2017, single-sided airing via windows, wind/stack), or 
- from occupancy-driven flow (simplified volumetric rate per floor area).

For more detail refers to [natural ventilation](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_ventilation/).

Due to the need to have profiles of occupancy and consumption of buildings for some uses, tables of profiles useful for evaluating, occupancy, lights, heating, cooling, internal gains have been implemented.
These tables are provided by ANNEX A of ISO EN 16798-1. 
In the tool they are available here: [Table](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/source/table_iso_16798_1.py)

## Input Quality check  **(New)**

The data provided before being used for the simulation are processed and evaluated to be considered fit for the simulation. This process includes a series of checks that allow to identify any potential errors. 
For more details refers to [Input Quality check](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_input_check/).

## Limitations

The library is developed with the intent of demonstrating specific elements of calculation procedures in the relevant standards. It is not intended to replace the regulations but to complement them, as the latter are essential for understanding the calculation. This library is meant to be used for demonstration and testing purposes and is therefore provided as open source, without protection against misuse or inappropriate use.

The information and views set out in this document are those of the authors and do not necessarily reflect the official opinion of the European Union.

## Getting Started

Install the latest version of the library:

```bash
pip install pybuildingenergy
```

## Building - System Inputs

- For building inputs refer to [Building Inputs](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_input/)
- For heating system inputs (EN 15316-1) refer to [Heating System Input](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_15316_input/)


## Example

**New examples will follow soon...**

## Contributing and Support

**Bug reports / Questions**  
If you encounter a bug, please create an issue detailing it. Provide steps to reproduce and a code snippet if possible.

**Code contributions**  
We welcome and appreciate contributions! Every contribution, no matter how small, makes a difference.

## License

- Free software: BSD 3-Clause License  
- Documentation: [pyBuildingEnergy Docs](https://eurac-eebgroup.github.io/pybuildingenergy-docs/)

## Author

- [Daniele Antonucci](https://www.eurac.edu/it/people/daniele-antonucci)
- [Ulrich Filippi Oberegger](https://www.eurac.edu/it/people/ulrich-filippi)
- [Olga Somova](https://www.eurac.edu/it/people/olga-somova)

## Acknowledgment

This work was carried out within European projects:
- **Infinite** — EU Horizon 2020 (grant agreement No. 958397)  
- **Moderate** — Horizon Europe (grant agreement No. 101069834)

DHW Calculation developed with data and methods from EPBCenter spreadsheet.

## References

- [EPB Center - Energy Performance of Buildings Directive (EPBD)](https://epb.center/epb-standards/the-energy-performance-of-buildings-directive-epbd/)
- [REHVA Journal - EN ISO 52000 family of standards](https://www.rehva.eu/rehva-journal/chapter/the-new-en-iso-52000-family-of-standards-to-assess-the-energy-performance-of-buildings-put-in-practice)
- [European Commission - Energy Performance of Buildings Directive](https://energy.ec.europa.eu/topics/energy-efficiency/energy-performance-buildings/energy-performance-buildings-directive_en)
- Directive (EU) 2024/1275 - Official Journal of the EU, May 8, 2024
- EN ISO 52010-1:2018 - External climatic conditions  
- EN ISO 52016-1:2018 - Energy needs for heating and cooling  
- EN ISO 52016-2:2018 - Explanation and justification of ISO 52016-1 and iso 52017-1
- EN 12831-3:2018 - DHW systems heat load and characterization  
- EN 15316-1:2018 - System energy requirements and efficiencies  
- EN 16798-7 & 16798-1 - Ventilation standards
