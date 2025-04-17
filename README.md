# utility-route-planner
The utility network needs to be expanded due to the energy transition. Finding a location for new infrastructure is no
easy feat considering the amount of involved design criteria. This research includes the creation of a software package
for automatic placement of utility network using a combination of geo-information and graph theory.

# Installation
To install the utility-route-designer package, use [Poetry](https://python-poetry.org/):
```bash
poetry install
```

# Usage
Running the main file will create utility routes for the five included cases in the `data/examples` folder. Optionally edit the configuration file `mcda_presets.py` to change the weights of the environmental criteria.
```bash
poetry run python main.py
```
The resulting geopackages and tifs are placed in the `data/processed` folder. View them in QGIS or similar GIS GUI.

Run tests using pytest:
```bash
poetry run python -m pytest tests/
```

# License
utility-route-designer is under: [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

The software is largely depended on data. Data incorporated in the example folder and their respective licenses are:
- BGT: [CC PDM 1.0](https://creativecommons.org/publicdomain/mark/1.0/deed.en) downloaded from [PDOK](https://www.nationaalgeoregister.nl/geonetwork/srv/dut/catalog.search#/metadata/e01e63cd-6b3d-4c58-b34e-8d343a3c264b).
- Natura2000: [CC PDM 1.0](https://creativecommons.org/publicdomain/mark/1.0/deed.en) downloaded from [PDOK](https://nationaalgeoregister.nl/geonetwork/srv/dut/catalog.search#/metadata/1601e160-91e8-4091-9aca-10294f819d42).
- Alliander asset information: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en) downloaded from ArcGIS Online: [gas](https://alliander.maps.arcgis.com/home/item.html?id=29b06805ca2b4d31bf82ad15f14d2392), [electricity](https://alliander.maps.arcgis.com/home/item.html?id=11b7bcf1b78b4462b91db0dff234cf78)

Citing
-------
t.b.d.
