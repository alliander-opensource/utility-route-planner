<!--
SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.

SPDX-License-Identifier: Apache-2.0
-->

# Utility Route Planner

This repository shares research on software for automatic placement of electricity cables using a combination of geo-information and graph theory.

The utility network needs to be expanded due to the energy transition. Finding a location for new infrastructure is no easy feat considering the amount of involved design criteria. 
This research includes the creation of a software package for automatic placement of utility network using a combination of geo-information and graph theory.

This research is being carried out at Alliander, a Dutch DSO, as part of Jelmar Versleijen's PhD with [Wagening University](https://www.wur.nl/en.htm). [Read more about research at Alliander](https://www.alliander.com/nl/alliander-en-open-research/).

Details on methodology can be found in publication: [Link to be added.]

# Goals for sharing

Our goal for sharing this software is to encourage research on utility route planning for distribution system operators.
Researchers or education can use the real-life use cases presented in this repository to test their own algorithms. The software is not intended for production use.

Remaining challenges solve are:

- Constraining the route to a maximum length.
- Giving alternative routes which are similar in costs (like seen in modern navigation systems).
- Realistic road crossings. Road crossings are typically done at a 90 degree angle through a process called pipe ramming.
- Alignment to existing infrastructure, resulting in a more "human-like" route.

# Installation

To install the utility-route-designer package, use Python 3.12 with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

# Usage

Running the main file will create utility routes for the five included cases in the `data/examples` folder. Optionally edit the configuration file `mcda_presets.py` to change the weights of the environmental criteria.

```bash
poetry run python main.py
```

The results are placed in the `data/processed` folder:

- mcda_output.gpkg: contains the environmental criteria as vectors used in creating the suitability raster
- benchmark_suitability_raster.vrt: the suitability raster / cost-surface used for the least-cost path analysis
- lcpa_results.gpkg: contains the generated route as linestring

View them in QGIS or similar GIS GUI:

![benchmark_results_overview.png](data/examples/benchmark_results_overview.png)
Run tests using pytest:

```bash
poetry run python -m pytest tests/
```

Expanding criteria included in the `mcda_presets.py` can be done by:

1. Adding a new class to the `criteria` folder.
2. Implementing the `get_suitability` method.
3. Adding the new class to the `mcda_presets.py` file. Set the group and weight of the new class.

# Support

If you have trouble installing, building, or using utility-route-planner, but you don't think you've encountered a genuine bug, you can ask a question in the Issues tab of the repository.
If you have an idea for a new feature or a recommendation for existing features or documentation, you can also propose it in the Issues tab.

## How to report a bug or a security vulnerability

This project manages bug and enhancement using the GitHub issue tracker.

# Contributing

Please read CODE_OF_CONDUCT and CONTRIBUTING, for details on the process for submitting pull requests to us.

# Project governance

The lead developer is responsible for reviewing contributions from the community and general direction of development.
When the community has grown to a size where it would be helpful, the project will set up independent community governance.

The lead developer is [Jelmar Versleijen](https://research.wur.nl/en/persons/jelmar-versleijen).

# License

utility-route-designer is under: [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

The software is largely dependent on data. Data is incorporated in the example folder and is licensed separately from the repo:

- BGT: [CC PDM 1.0](https://creativecommons.org/publicdomain/mark/1.0/deed.en) downloaded from [PDOK](https://www.nationaalgeoregister.nl/geonetwork/srv/dut/catalog.search#/metadata/e01e63cd-6b3d-4c58-b34e-8d343a3c264b)
- Natura2000: [CC PDM 1.0](https://creativecommons.org/publicdomain/mark/1.0/deed.en) downloaded from [PDOK](https://nationaalgeoregister.nl/geonetwork/srv/dut/catalog.search#/metadata/1601e160-91e8-4091-9aca-10294f819d42)
- Alliander asset information: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en) downloaded from ArcGIS Online: [gas](https://alliander.maps.arcgis.com/home/item.html?id=29b06805ca2b4d31bf82ad15f14d2392), [electricity](https://alliander.maps.arcgis.com/home/item.html?id=11b7bcf1b78b4462b91db0dff234cf78)

Citing
-------
t.b.d.
