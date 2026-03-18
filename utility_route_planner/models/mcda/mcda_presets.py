# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from settings import Config
import geopandas as gpd

from utility_route_planner.models.mcda.vector_preprocessing.begroeidterreindeel import BegroeidTerreindeel
from utility_route_planner.models.mcda.vector_preprocessing.existing_substations import ExistingSubstations
from utility_route_planner.models.mcda.vector_preprocessing.wegdeel import Wegdeel
from utility_route_planner.models.mcda.vector_preprocessing.excluded_area import ExcludedArea
from utility_route_planner.models.mcda.vector_preprocessing.existing_utilities import ExistingUtilities
from utility_route_planner.models.mcda.vector_preprocessing.kunstwerkdeel import Kunstwerkdeel
from utility_route_planner.models.mcda.vector_preprocessing.onbegroeid_terreindeel import OnbegroeidTerreindeel
from utility_route_planner.models.mcda.vector_preprocessing.ondersteunend_waterdeel import OndersteunendWaterdeel
from utility_route_planner.models.mcda.vector_preprocessing.ondersteunend_wegdeel import OndersteunendWegdeel
from utility_route_planner.models.mcda.vector_preprocessing.overig_bouwwerk import OverigBouwwerk
from utility_route_planner.models.mcda.vector_preprocessing.pand import Pand
from utility_route_planner.models.mcda.vector_preprocessing.protected_area import ProtectedArea
from utility_route_planner.models.mcda.vector_preprocessing.small_above_ground_obstacles import (
    SmallAboveGroundObstacles,
)
from utility_route_planner.models.mcda.vector_preprocessing.vegetation_object import VegetationObject
from utility_route_planner.models.mcda.vector_preprocessing.waterdeel import Waterdeel

preset_collection = {
    "preset_benchmark_raw": {
        "general": {
            "description": "Preset used for benchmark results.",
            "prefix": "bm_",
            "final_raster_name": "benchmark_suitability_raster",
            "project_area_geometry": gpd.read_file(
                Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA
            )
            .iloc[0]
            .geometry,
        },
        # BGT explanation: https://docs.geostandaarden.nl/imgeo/catalogus/bgt/#attributen-en-associaties
        "criteria": {
            "waterdeel": {
                "description": "Information on water: https://geonovum.github.io/IMGeo-objectenhandboek/waterdeel",
                "layer_names": ["bgt_waterdeel_V"],
                "preprocessing_function": Waterdeel(),
                "group": "a",
                "columns_to_reclassify": ["class", "plus-type"],
                "weight_values": {
                    # Column "class"
                    "greppel, droge sloot": 13,
                    "waterloop": 126,
                    "watervlakte": 126,
                    "zee": 126,
                    # Column "plus-type"
                    "rivier": 126,
                    "sloot": 21,
                    "kanaal": 126,
                    "beek": 126,
                    "gracht": 126,
                    "bron": 126,
                    "haven": 126,
                    "meer, plas, ven, vijver": 21,
                },
                "geometry_values": {"zee": 20},
            },
            "ondersteunend_waterdeel": {
                "description": "Information on complementary bits related to water: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/ondersteunendwaterdeel",
                "layer_names": ["bgt_ondersteunendwaterdeel_V"],
                "preprocessing_function": OndersteunendWaterdeel(),
                "group": "a",
                "columns_to_reclassify": ["class"],
                "weight_values": {
                    # class
                    "oever, slootkant": 13,
                    "slik": 13,
                },
            },
            "wegdeel": {
                "description": "Information on roads: https://geonovum.github.io/IMGeo-objectenhandboek/wegdeel",
                "layer_names": ["bgt_wegdeel_V"],
                "preprocessing_function": Wegdeel(),
                "group": "a",
                "columns_to_reclassify": ["function", "surfaceMaterial"],
                "weight_values": {
                    # Column "function"
                    "baan voor vliegverkeer": 126,
                    "fietspad": 4,
                    "inrit": 5,
                    "OV-baan": 5,
                    "overweg": 120,
                    "parkeervlak": 3,
                    "rijbaan autosnelweg": 126,  # Motorway
                    "rijbaan autoweg": 126,  # Motorway
                    "rijbaan lokale weg": 6,
                    "rijbaan regionale weg": 9,  # Provincial road
                    "ruiterpad": 3,
                    "spoorbaan": 126,
                    "voetgangersgebied": 3,
                    "voetpad": 3,
                    "voetpad op trap": 3,
                    "woonerf": 3,
                    # Column surfaceMaterial
                    "gesloten verharding": 33,
                    "half verhard": 4,
                    "onverhard": 3,
                    "open verharding": 6,
                    "transitie": 3,
                },
            },
            "ondersteunend_wegdeel": {
                "description": "Information on complementary related to roads: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/ondersteunendwegdeel",
                "layer_names": ["bgt_ondersteunendwegdeel_V"],
                "preprocessing_function": OndersteunendWegdeel(),
                "group": "a",
                "columns_to_reclassify": ["function", "surfaceMaterial"],
                "weight_values": {
                    # function
                    "berm": 2,
                    "verkeerseiland": 126,
                    # surfaceMaterial
                    "gesloten verharding": 33,
                    "groenvoorziening": 3,
                    "half verhard": 4,
                    "onverhard": 2,
                    "open verharding": 6,
                },
            },
            "onbegroeid_terreindeel": {
                "description": "Information on non-lush terrain: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/onbegroeidterreindeel",
                "layer_names": ["bgt_onbegroeidterreindeel_V"],
                "preprocessing_function": OnbegroeidTerreindeel(),
                "group": "a",
                "columns_to_reclassify": ["bgt-fysiekVoorkomen"],
                "weight_values": {
                    # bgt_fysiekvoorkomen
                    "erf": 76,  # is now used as an approximation if ground is public or privately owned.
                    "gesloten verharding": 33,
                    "half verhard": 4,
                    "onverhard": 2,
                    "open verharding": 6,
                    "zand": 2,
                },
            },
            "begroeid_terreindeel": {
                "description": "Information on lush terrain: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/begroeidterreindeel",
                "layer_names": ["bgt_begroeidterreindeel_V"],
                "preprocessing_function": BegroeidTerreindeel(),
                "group": "a",
                "columns_to_reclassify": ["class", "plus-fysiekVoorkomen"],
                "weight_values": {
                    # class
                    "boomteelt": 76,  # implies private property
                    "bouwland": 76,  # implies private property
                    "duin": 67,
                    "fruitteelt": 76,  # implies private property
                    "gemengd bos": 67,
                    "grasland agrarisch": 31,  # implies private property, but easy to cross and repair afterward.
                    "grasland overig": 31,  # implies private property, but easy to cross and repair afterward.
                    "groenvoorziening": 3,  # implies public property
                    "heide": 67,
                    "houtwal": 67,
                    "kwelder": 67,
                    "loofbos": 67,
                    "moeras": 67,
                    "naaldbos": 67,
                    "rietland": 67,
                    "struiken": 3,  # implies public property
                    # plus-fysiekVoorkomenWegdeel
                    "akkerbouw": 76,  # implies private property
                    "bodembedekkers": 3,
                    "bollenteelt": 76,  # implies private property
                    "bosplantsoen": 10,  # implies public property within build-up area with (sparsely placed) trees
                    "braakliggend": 76,  # implies private property
                    "gesloten duinvegetatie": 10,
                    "gras- en kruidachtigen": 3,
                    "griend en hakhout": 67,
                    "heesters": 3,
                    "hoogstam boomgaarden": 76,  # implies private property
                    "klein fruit": 76,  # implies private property
                    "laagstam boomgaarden": 76,  # implies private property
                    "open duinvegetatie": 10,
                    "planten": 3,
                    "struikrozen": 3,
                    "vollegrondsteelt": 76,  # implies private property
                    "wijngaarden": 76,  # implies private property, hard to "repair" if damaged.
                },
            },
            "pand": {
                "description": "Information on buildings: https://geonovum.github.io/IMGeo-objectenhandboek/pand",
                "layer_names": ["bgt_pand_V"],
                "preprocessing_function": Pand(),
                "group": "a",
                "weight_values": {"pand": 126},
            },
            "overig_bouwwerk": {
                "description": "Information on miscellaneous buildings: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/overigbouwwerk",
                "layer_names": ["bgt_overigbouwwerk_V"],
                "preprocessing_function": OverigBouwwerk(),
                "group": "b",
                "columns_to_reclassify": ["bgt-type"],
                "weight_values": {
                    # bgt-type
                    "functie": 1,
                    "bassin": 1,
                    "bezinkbak": 1,
                    "lage trafo": 1,  # Includes Alliander substations, but not always. Is overwritten by existing_substation.
                    "niet-bgt": 1,  # Delete these records if they exist.
                    "open loods": 1,
                    "opslagtank": 1,
                    "overkapping": 1,
                    "windturbine": 126,
                },
            },
            "kunstwerkdeel": {
                "description": "Information on civil engineering structures: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/kunstwerkdeel",
                "layer_names": ["bgt_kunstwerkdeel_V"],
                "preprocessing_function": Kunstwerkdeel(),
                "group": "a",
                "columns_to_reclassify": ["bgt-type"],
                "weight_values": {
                    # bgt-type
                    "gemaal": 126,
                    "hoogspanningsmast": 126,
                    "niet-bgt": 1,  # Delete these records if they exist.
                    "perron": 126,
                    "sluis": 126,
                    "steiger": 126,
                    "strekdam": 126,
                    "stuw": 126,
                },
            },
            "small_above_ground_obstacles": {
                "description": "Information on small above ground obstacles: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/scheiding, "
                "https://geonovum.github.io/IMGeo-objectenhandboek/bak,"
                "https://geonovum.github.io/IMGeo-objectenhandboek/bord,"
                "https://geonovum.github.io/IMGeo-objectenhandboek/kast,"
                "https://geonovum.github.io/IMGeo-objectenhandboek/mast,"
                "https://geonovum.github.io/IMGeo-objectenhandboek/paal,"
                "https://geonovum.github.io/IMGeo-objectenhandboek/put,"
                "https://geonovum.github.io/IMGeo-objectenhandboek/sensor,"
                "https://geonovum.github.io/IMGeo-objectenhandboek/straatmeubilair,",
                # The order of layers matter here, first two must be bgt_scheiding.
                "layer_names": [
                    "bgt_scheiding_V",
                    "bgt_scheiding_L",
                    "bgt_bak_P",
                    "bgt_bord_P",
                    "bgt_kast_P",
                    "bgt_mast_P",
                    "bgt_paal_P",
                    "bgt_put_P",
                    "bgt_sensor_P",
                    "bgt_straatmeubilair_P",
                ],
                "preprocessing_function": SmallAboveGroundObstacles(),
                "group": "b",
                "columns_to_reclassify": ["bgt-type", "plus-type"],
                "weight_values": {
                    # scheiding: bgt_type
                    "damwand": 126,
                    "muur": 76,
                    "kademuur": 126,
                    "geluidsscherm": 76,
                    "hek": 13,
                    "niet-bgt": 1,  # Delete these records if they exist.
                    "walbescherming": 126,
                    # bak: plus_type
                    "afval apart plaats": 76,  # These containers are underground.
                    "afvalbak": 4,
                    "bloembak": 4,
                    "container": 4,
                    "drinkbak": 4,
                    "zand- / zoutbak": 4,
                    # bord: plus_type
                    "dynamische snelheidsindicator": 4,
                    "informatiebord": 4,
                    "plaatsnaambord": 4,
                    "reclamebord": 4,
                    "scheepvaartbord": 4,
                    "straatnaambord": 4,
                    "verkeersbord": 4,
                    "verklikker transportleiding": 4,
                    "waarschuwingshek": 4,
                    "wegwijzer": 4,
                    # kast: plus_type
                    "CAI-kast": 4,
                    "elektrakast": 4,
                    "gaskast": 4,
                    "GMS kast": 4,
                    "openbare verlichtingkast": 4,
                    "rioolkast": 4,
                    "telecom kast": 4,
                    "telkast": 4,
                    "verkeersregelinstallatiekast": 4,
                    # mast: plus_type
                    "bovenleidingmast": 4,
                    "laagspanningsmast": 4,
                    "radarmast": 4,
                    "straalzender": 4,
                    "zendmast": 4,
                    # paal: plus_type
                    "afsluitpaal": 4,
                    "dijkpaal": 4,
                    "drukknoppaal": 4,
                    "grensmarkering": 4,
                    "haltepaal": 4,
                    "hectometerpaal": 4,
                    "lichtmast": 4,
                    "poller": 4,
                    "portaal": 4,
                    "praatpaal": 4,
                    "sirene": 4,
                    "telpaal": 4,
                    "verkeersbordpaal": 4,
                    "verkeersregelinstallatiepaal": 4,
                    "vlaggenmast": 4,
                    # put: plus_type
                    "benzine- / olieput": 4,
                    "brandkraan / -put": 4,
                    "drainageput": 4,
                    "gasput": 4,
                    "inspectie- / rioolput": 4,
                    "kolk": 4,
                    "waterleidingput": 4,
                    # sensor:
                    "detectielus": 4,
                    "camera": 4,
                    "debietmeter": 4,
                    "flitser": 4,
                    "GMS sensor": 4,
                    "hoogtedetectieapparaat": 4,
                    "lichtcel": 4,
                    "radar detector": 4,
                    "waterstandmeter": 4,
                    "weerstation": 4,
                    "windmeter": 4,
                    # straatmeubilair:
                    "abri": 4,
                    "bank": 4,
                    "betaalautomaat": 4,
                    "bolder": 4,
                    "brievenbus": 4,
                    "fietsenkluis": 4,
                    "fietsenrek": 4,
                    "fontein": 51,
                    "herdenkingsmonument": 126,
                    "kunstobject": 51,
                    "lichtpunt": 4,
                    "openbaar toilet": 51,
                    "parkeerbeugel": 4,
                    "picknicktafel": 4,
                    "reclamezuil": 4,
                    "slagboom": 4,
                    "speelvoorziening": 4,
                    "telefooncel": 4,
                    # Value occurs on all these tables, remove if present.
                    "waardeOnbekend": 1,
                },
            },
            "vegetation_object": {
                "description": "Information on seperate vegetation objects such as trees or hedges: "
                "https://geonovum.github.io/IMGeo-objectenhandboek/vegetatieobject",
                "layer_names": ["bgt_vegetatieobject_P", "bgt_vegetatieobject_V"],
                "preprocessing_function": VegetationObject(),
                "group": "b",
                "columns_to_reclassify": ["plus-type"],
                "weight_values": {
                    # plus_type
                    "haag": 3,
                    "boom": 10,
                    "waardeOnbekend": 1,  # Delete these records if they exist.
                },
                "geometry_values": {"boom": 5},
            },
            "protected_area": {
                # https://geonovum.github.io/IMGeo-objectenhandboek/functioneelgebied
                "description": "Protected area such as dykes and nature which may have additional rules or policies.",
                "layer_names": ["bgt_functioneelgebied_V", "natura2000"],
                "preprocessing_function": ProtectedArea(),
                "group": "b",
                "columns_to_reclassify": ["type"],
                "weight_values": {
                    # bgt_type
                    "kering": 10,  # Dykes, all other features of bgt_functioneelgebied are removed.
                    "natura2000": 10,
                    # Possible extra (paid) data which is relevant to utility routing.
                    # "Aardkundig_monument": 1,
                    # "Archeologisch_monument": 1,
                    # "niet_gesprongen_explosieven_wo2": 1,
                    # "brosse_leidingen": 1,
                    # "verontreinigde_grond": 1,
                    # "groene_en_rijksmonumenten": 1,
                },
            },
            "existing_utilities": {
                # TenneT, Alliander, Gasunie.
                "description": "Existing utility assets for gas and electricity.",
                "layer_names": [
                    "hoogspanningskabel_bovengronds",
                    "hoogspanningskabel_ondergronds",
                    "gasunie_leidingen",
                    "alliander_stationsterrein",
                ],
                "preprocessing_function": ExistingUtilities(),
                "group": "b",
                "columns_to_reclassify": ["asset_type"],
                "weight_values": {
                    "hoogspanning_bovengronds": 4,  # TenneT & Alliander combined.
                    "hoogspanning_ondergronds": 51,  # TenneT & Alliander combined.
                    "gasunie_leidingen": 51,
                    "alliander_stationsterrein": -126,  # Only the larger (>30m2) areas are included.
                },
                "geometry_values": {
                    "hoogspanning_bovengronds_buffer": 5,
                    "hoogspanning_ondergronds_buffer": 5,
                    "gasunie_leidingen_buffer": 5,
                },
            },
            "existing_substations": {
                "description": "Existing electricity substations.",
                "layer_names": ["alliander_middenspanningsstation"],
                "preprocessing_function": ExistingSubstations(),
                "group": "a",
                "weight_values": {
                    "alliander_middenspanningsstation": 126,  # Building footprint of a (sub)station.
                },
            },
            "excluded_area": {
                "description": "Area to exclude were no utility network can be placed, manually provided.",
                "layer_names": ["area_to_exclude"],
                "preprocessing_function": ExcludedArea(),
                "group": "c",
                "weight_values": {
                    # The weight value does not matter as group c determines that all features are excluded.
                    "constraint": 1,
                },
            },
        },
    },
}
