# pyBuildingEnergy

![pyBuildingEnergy Logo](https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/Logo_pyBuild.png)

## Citation
Please cite us if you use the library.

[![DOI](https://zenodo.org/badge/761715706.svg)](https://zenodo.org/doi/10.5281/zenodo.10887919)

## Features

The new EPBD recast provides an update on building performance assessment through a methodology that must take into account various aspects such as the thermal characteristics of the building, the use of energy from renewable sources, building automation and control systems, ventilation, cooling, energy recovery, etc.

The methodology should represent the actual operating conditions, allow for the use of measured energy for accuracy and comparability purposes, and be based on hourly or sub-hourly intervals that take into account the variable conditions significantly impacting the operation and performance of the system, as well as internal conditions.

**pyBuildingEnergy** aims to provide an assessment of building performance both in terms of energy and comfort. In this initial release, it is possible to assess the energy performance of the building using ISO 52106-1:2018. Additional modules will be added for a more comprehensive evaluation of performance, assessing ventilation, renewable energies, systems, etc.

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

## Weather Data

The tool can use weather data coming from 2 main sources:

- PVGIS API ([link](https://re.jrc.ec.europa.eu/pvg_tools/en/)) - PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM
- `.epw` file from [Ladybug Tools EPWMap](https://www.ladybug.tools/epwmap/)

More details in the example folder.

## Domestic Hot Water - DHW

- [x] Calculation of volume and energy need for domestic hot water according to ISO 12831-3.
- [ ] Assessment of thermal load based on the type of DHW system.

## Primary Energy - Heating System

The ISO EN 15316 series covers the calculation method for system energy requirements and system efficiencies. This family of standards is an integral part of the EPB set and covers:

### ISO EN 15316 Modular Structure

- [x] ISO EN 15316-1: General and expression of energy performance (Modules M3-1, M3-4, M3-9, M8-1, M8-4)
- [ ] ISO EN 15316-2: Emission systems (heating and cooling)
- [ ] ISO EN 15316-3: Distribution systems (DHW, heating, cooling)
- [ ] ISO EN 15316-4-X: Heat generation systems:
  - 4-1: Combustion boilers
  - 4-2: Heat pumps
  - 4-3: Solar thermal and photovoltaic systems
  - 4-4: Cogeneration systems
  - 4-5: District heating
  - 4-7: Biomass
- [ ] ISO EN 15316-5: Storage systems

For space heating, applicable standards include ISO EN 15316-1, ISO EN 15316-2-1, ISO EN 15316-2-3 and the appropriate parts of ISO EN 15316-4 depending on the system type, including losses and control aspects.

## Limitations

The library is developed with the intent of demonstrating specific elements of calculation procedures in the relevant standards. It is not intended to replace the regulations but to complement them, as the latter are essential for understanding the calculation. This library is meant to be used for demonstration and testing purposes and is therefore provided as open source, without protection against misuse or inappropriate use.

The information and views set out in this document are those of the authors and do not necessarily reflect the official opinion of the European Union.

The calculation is currently aimed at single-zone buildings with ground floor. The evaluation of multi-zone buildings is under evaluation.

## Getting Started

Install the latest version of the library:

```bash
pip install pybuildingenergy
```

## Building - System Inputs

- For building inputs refer to [Building Inputs](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_input/)
- For heating system inputs (ISO EN 15316-1) refer to [Heating System Input](https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_15316_input/)

## Documentation

Check our documentation [here](https://eurac-eebgroup.github.io/pybuildingenergy-docs/).

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
- EN ISO 12831-3:2018 - DHW systems heat load and characterization  
- EN ISO 15316-1:2018 - System energy requirements and efficiencies  
- EN ISO 16798-7 & 16798-1 - Ventilation standards
