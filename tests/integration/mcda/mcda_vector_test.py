# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from settings import Config
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.mcda.vector_preprocessing.begroeidterreindeel import BegroeidTerreindeel
from utility_route_planner.models.mcda.vector_preprocessing.excluded_area import ExcludedArea
from utility_route_planner.models.mcda.vector_preprocessing.existing_substations import ExistingSubstations
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
from utility_route_planner.models.mcda.mcda_presets import preset_collection
from utility_route_planner.models.mcda.vector_preprocessing.wegdeel import Wegdeel
from utility_route_planner.util.write import reset_geopackage
import geopandas as gpd
import shapely


@pytest.fixture
def setup_clean_start(monkeypatch):
    reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)


@pytest.mark.usefixtures("setup_clean_start")
class TestVectorPreprocessing:
    def test_process_single_vector_criteria_waterdeel(self):
        # Filter the preset to only 1 criterion.
        preset_to_load = {
            "general": preset_collection["preset_benchmark_raw"]["general"],
            "criteria": {"waterdeel": preset_collection["preset_benchmark_raw"]["criteria"]["waterdeel"]},
        }
        mcda_engine = McdaCostSurfaceEngine(
            preset_to_load,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )
        mcda_engine.preprocess_vectors()

    def test_process_all_vectors(self):
        mcda_engine = McdaCostSurfaceEngine(
            Config.RASTER_PRESET_NAME_BENCHMARK,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )
        mcda_engine.preprocess_vectors()

    def test_process_waterdeel(self):
        weight_values = {
            # Column "class"
            "greppel, droge sloot": -10,
            "waterloop": 1,
            "watervlakte": 2,
            "zee": 3,
            # Column "plus-type"
            "rivier": 40,
            "sloot": 50,
            "kanaal": 60,
            "beek": 70,
            "gracht": 80,
            "bron": 90,
            "haven": 100,
            "meer, plas, ven, vijver": 110,
        }
        input_gdf = gpd.GeoDataFrame(
            [
                ["greppel, droge sloot", "WaardeOnbekend", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waterloop", "WaardeOnbekend", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waterloop", "rivier", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waterloop", "sloot", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waterloop", "kanaal", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waterloop", "beek", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waterloop", "gracht", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waterloop", "bron", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["watervlakte", "haven", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["watervlakte", "meer, plas, ven, vijver", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["zee", "WaardeOnbekend", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
            ],
            columns=["class", "plus-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )

        reclassified_gdf = Waterdeel._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([-10, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_2"],
            pd.Series(["WaardeOnbekend", "WaardeOnbekend", 40, 50, 60, 70, 80, 90, 100, 110, "WaardeOnbekend"]),
            check_names=False,
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([-10, 1, 40, 50, 60, 70, 80, 90, 100, 110, 3]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )

        buffered_gdf = Waterdeel._update_geometry_values(reclassified_gdf, {"zee": 20})
        assert buffered_gdf.iloc[:10].area.round(1).unique().tolist() == [1.0]
        assert buffered_gdf.iloc[[10]].area.round(1).tolist() == [1335.6]

    def test_begroeid_terreindeel(self):
        weight_values = {
            # bgt_fysiekvoorkomen
            "boomteelt": 1,
            "bouwland": 2,
            "duin": 3,
            "fruitteelt": 4,
            "gemengd bos": 5,
            "grasland agrarisch": 6,
            "grasland overig": 7,
            "groenvoorziening": 8,
            "heide": 9,
            "houtwal": 10,
            "kwelder": 11,
            "loofbos": 12,
            "moeras": 13,
            "naaldbos": 14,
            "rietland": 15,
            "struiken": 16,
            # plus_fysiekvoorkomen
            "akkerbouw": 17,
            "bodembedekkers": 18,
            "bollenteelt": 19,
            "bosplantsoen": 20,
            "braakliggend": 21,
            "gesloten duinvegetatie": 22,
            "gras- en kruidachtigen": 23,
            "griend en hakhout": 24,
            "heesters": 25,
            "hoogstam boomgaarden": 26,
            "klein fruit": 27,
            "laagstam boomgaarden": 28,
            "open duinvegetatie": 29,
            "planten": 30,
            "struikrozen": 31,
            "vollegrondsteelt": 32,
            "wijngaarden": 33,
        }
        input_gdf = gpd.GeoDataFrame(
            [
                ["boomteelt", "waardeOnbekend", shapely.Polygon()],
                ["bouwland", "waardeOnbekend", shapely.Polygon()],
                ["duin", "waardeOnbekend", shapely.Polygon()],
                ["fruitteelt", "waardeOnbekend", shapely.Polygon()],
                ["gemengd bos", "waardeOnbekend", shapely.Polygon()],
                ["grasland agrarisch", "waardeOnbekend", shapely.Polygon()],
                ["grasland overig", "waardeOnbekend", shapely.Polygon()],
                ["groenvoorziening", "waardeOnbekend", shapely.Polygon()],
                ["heide", "waardeOnbekend", shapely.Polygon()],
                ["houtwal", "waardeOnbekend", shapely.Polygon()],
                ["kwelder", "waardeOnbekend", shapely.Polygon()],
                ["loofbos", "waardeOnbekend", shapely.Polygon()],
                ["moeras", "waardeOnbekend", shapely.Polygon()],
                ["naaldbos", "waardeOnbekend", shapely.Polygon()],
                ["rietland", "waardeOnbekend", shapely.Polygon()],
                ["struiken", "waardeOnbekend", shapely.Polygon()],
                # Use placeholders for class as they will be overwritten.
                ["loofbos", "akkerbouw", shapely.Polygon()],
                ["loofbos", "bodembedekkers", shapely.Polygon()],
                ["loofbos", "bollenteelt", shapely.Polygon()],
                ["loofbos", "bosplantsoen", shapely.Polygon()],
                ["loofbos", "braakliggend", shapely.Polygon()],
                ["loofbos", "gesloten duinvegetatie", shapely.Polygon()],
                ["loofbos", "gras- en kruidachtigen", shapely.Polygon()],
                ["loofbos", "griend en hakhout", shapely.Polygon()],
                ["loofbos", "heesters", shapely.Polygon()],
                ["loofbos", "hoogstam boomgaarden", shapely.Polygon()],
                ["loofbos", "klein fruit", shapely.Polygon()],
                ["loofbos", "laagstam boomgaarden", shapely.Polygon()],
                ["loofbos", "open duinvegetatie", shapely.Polygon()],
                ["loofbos", "planten", shapely.Polygon()],
                ["loofbos", "struikrozen", shapely.Polygon()],
                ["loofbos", "vollegrondsteelt", shapely.Polygon()],
                ["loofbos", "wijngaarden", shapely.Polygon()],
            ],
            columns=["class", "plus-fysiekVoorkomen", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = BegroeidTerreindeel._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                ]
            ),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_2"],
            pd.Series(
                [
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    "waardeOnbekend",
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                ]
            ),
            check_names=False,
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                ]
            ),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )

    def test_kunstwerkdeel(self):
        weight_values = {
            # bgt_type
            "gemaal": 1,
            "hoogspanningsmast": 2,
            "niet-bgt": 3,
            "perron": 4,
            "sluis": 5,
            "steiger": 6,
            "strekdam": 7,
            "stuw": 8,
        }
        input_gdf = gpd.GeoDataFrame(
            [
                ["gemaal", shapely.Polygon()],
                ["hoogspanningsmast", shapely.Polygon()],
                ["niet-bgt", shapely.Polygon()],
                ["perron", shapely.Polygon()],
                ["sluis", shapely.Polygon()],
                ["steiger", shapely.Polygon()],
                ["strekdam", shapely.Polygon()],
                ["stuw", shapely.Polygon()],
            ],
            columns=["bgt-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )

        reclassified_gdf = Kunstwerkdeel._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([1, 2, 4, 5, 6, 7, 8]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 2, 4, 5, 6, 7, 8]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        assert "niet-bgt" not in reclassified_gdf["bgt-type"]

    def test_onbegroeid_terreindeel(self):
        weight_values = {
            # bgt_fysiekvoorkomen
            "erf": 1,
            "gesloten verharding": 2,
            "half verhard": 3,
            "onverhard": 4,
            "open verharding": 5,
            "zand": 6,
        }
        input_gdf = gpd.GeoDataFrame(
            [
                ["erf", shapely.Polygon()],
                ["gesloten verharding", shapely.Polygon()],
                ["half verhard", shapely.Polygon()],
                ["onverhard", shapely.Polygon()],
                ["open verharding", shapely.Polygon()],
                ["zand", shapely.Polygon()],
            ],
            columns=["bgt-fysiekVoorkomen", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )

        reclassified_gdf = OnbegroeidTerreindeel._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([1, 2, 3, 4, 5, 6]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 2, 3, 4, 5, 6]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )

    def test_ondersteunend_waterdeel(self):
        weight_values = {
            # bgt_type
            "oever, slootkant": 1,
            "slik": 2,
        }
        input_gdf = gpd.GeoDataFrame(
            [
                ["oever, slootkant", shapely.Polygon()],
                ["slik", shapely.Polygon()],
            ],
            columns=["class", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )

        reclassified_gdf = OndersteunendWaterdeel._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([1, 2]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 2]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )

    def test_ondersteunend_wegdeel(self):
        weight_values = {
            # bgt_functie
            "berm": 1,
            "verkeerseiland": 2,
            # bgt_fysiekvoorkomen
            "gesloten verharding": 3,
            "groenvoorziening": 4,
            "half verhard": 5,
            "onverhard": 6,
            "open verharding": 7,
        }
        input_gdf = gpd.GeoDataFrame(
            [
                ["berm", "waardeOnbekend", shapely.Polygon()],
                ["verkeerseiland", "waardeOnbekend", shapely.Polygon()],
                # Use placeholders for class as they will be overwritten.
                ["berm", "groenvoorziening", shapely.Polygon()],
                ["verkeerseiland", "gesloten verharding", shapely.Polygon()],
                ["verkeerseiland", "half verhard", shapely.Polygon()],
                ["verkeerseiland", "onverhard", shapely.Polygon()],
                ["verkeerseiland", "open verharding", shapely.Polygon()],
            ],
            columns=["function", "surfaceMaterial", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = OndersteunendWegdeel._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([1, 2, 1, 2, 2, 2, 2]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_2"],
            pd.Series(["waardeOnbekend", "waardeOnbekend", 4, 3, 5, 6, 7]),
            check_names=False,
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 2, 4, 3, 5, 6, 7]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
        )

    def test_overig_bouwwerk(self):
        weight_values = {
            # bgt_type
            "functie": 1,
            "bassin": 2,
            "bezinkbak": 3,
            "lage trafo": 4,
            "niet-bgt": 5,
            "open loods": 6,
            "opslagtank": 7,
            "overkapping": 8,
            "windturbine": 9,
        }
        input_gdf = gpd.GeoDataFrame(
            [
                ["functie", shapely.Polygon()],
                ["bassin", shapely.Polygon()],
                ["bezinkbak", shapely.Polygon()],
                ["lage trafo", shapely.Polygon()],
                ["niet-bgt", shapely.Polygon()],
                ["open loods", shapely.Polygon()],
                ["opslagtank", shapely.Polygon()],
                ["overkapping", shapely.Polygon()],
                ["windturbine", shapely.Polygon()],
            ],
            columns=["bgt-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )

        reclassified_gdf = OverigBouwwerk._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([1, 2, 3, 4, 6, 7, 8, 9]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 2, 3, 4, 6, 7, 8, 9]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        assert "niet-bgt" not in reclassified_gdf["bgt-type"]

    def test_pand(self):
        weight_values = {"pand": 1}
        input_gdf = gpd.GeoDataFrame(
            [
                ["placeholder", shapely.Polygon()],
            ],
            columns=["placeholder", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )

        reclassified_gdf = Pand._set_suitability_values(input_gdf, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )

    def test_protected_area(self):
        weight_values = {
            # bgt_type
            "kering": 1,  # Dykes, delete all other records
            "natura2000": 2,
        }
        input_gdf_bgt = gpd.GeoDataFrame(
            [
                ["kering", shapely.Polygon()],
                ["should_be_deleted", shapely.Polygon()],
                ["should_be_deleted_too", shapely.Polygon()],
            ],
            columns=["bgt-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        input_gdf_natura2000 = gpd.GeoDataFrame(
            [
                ["some_protected_area", 1, "test", shapely.Polygon()],
            ],
            columns=["naamN2K", "nr", "beschermin", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = ProtectedArea._set_suitability_values([input_gdf_bgt, input_gdf_natura2000], weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["type"],
            pd.Series(["kering", "natura2000"]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 2]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )

    def test_small_above_ground_obstacles(self):
        weight_values = {
            # scheiding: bgt_type
            "damwand": 1,
            "muur": 2,
            "kademuur": 3,
            "geluidsscherm": 4,
            "hek": 5,
            "niet-bgt": 6,  # Delete these records if they exist.
            "walbescherming": 7,
            # bak: plus_type
            "afval apart plaats": 8,
            "afvalbak": 9,
            "bloembak": 10,
            "container": 11,
            "drinkbak": 12,
            "zand- / zoutbak": 13,
            # bord: plus_type
            "dynamische snelheidsindicator": 14,
            "informatiebord": 15,
            "plaatsnaambord": 16,
            "reclamebord": 17,
            "scheepvaartbord": 18,
            "straatnaambord": 19,
            "verkeersbord": 20,
            "verklikker transportleiding": 21,
            "waarschuwingshek": 22,
            "wegwijzer": 23,
            # kast: plus_type
            "CAI-kast": 24,
            "elektrakast": 25,
            "gaskast": 26,
            "GMS kast": 27,
            "openbare verlichtingkast": 28,
            "rioolkast": 29,
            "telecom kast": 30,
            "telkast": 31,
            "verkeersregelinstallatiekast": 32,
            # mast: plus_type
            "bovenleidingmast": 33,
            "laagspanningsmast": 34,
            "radarmast": 35,
            "straalzender": 36,
            "zendmast": 37,
            # paal: plus_type
            "afsluitpaal": 38,
            "dijkpaal": 39,
            "drukknoppaal": 40,
            "grensmarkering": 41,
            "haltepaal": 42,
            "hectometerpaal": 43,
            "lichtmast": 44,
            "poller": 44,
            "portaal": 45,
            "praatpaal": 46,
            "sirene": 47,
            "telpaal": 48,
            "verkeersbordpaal": 49,
            "verkeersregelinstallatiepaal": 50,
            "vlaggenmast": 51,
            # put: plus_type
            "benzine- / olieput": 52,
            "brandkraan / -put": 53,
            "drainageput": 54,
            "gasput": 55,
            "inspectie- / rioolput": 56,
            "kolk": 57,
            "waterleidingput": 58,
            # sensor:
            "detectielus": 59,
            "camera": 59,
            "debietmeter": 60,
            "flitser": 61,
            "GMS sensor": 62,
            "hoogtedetectieapparaat": 63,
            "lichtcel": 64,
            "radar detector": 65,
            "waterstandmeter": 66,
            "weerstation": 67,
            "windmeter": 68,
            # straatmeubilair:
            "abri": 69,
            "bank": 70,
            "betaalautomaat": 71,
            "bolder": 72,
            "brievenbus": 72,
            "fietsenkluis": 73,
            "fietsenrek": 74,
            "fontein": 75,
            "herdenkingsmonument": 76,
            "kunstobject": 77,
            "lichtpunt": 78,
            "openbaar toilet": 79,
            "parkeerbeugel": 80,
            "picknicktafel": 81,
            "reclamezuil": 82,
            "slagboom": 83,
            "speelvoorziening": 84,
            "telefooncel": 85,
            # Value occurs on all these tables, remove if present.
            "waardeOnbekend": 1,
        }
        bgt_scheiding_1 = gpd.GeoDataFrame(
            [
                ["damwand", shapely.Polygon()],
                ["muur", shapely.Polygon()],
                ["kademuur", shapely.Polygon()],
                ["geluidsscherm", shapely.Polygon()],
                ["niet-bgt", shapely.Polygon()],
            ],
            columns=["bgt-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        bgt_scheiding_2 = gpd.GeoDataFrame(
            [
                ["hek", shapely.Polygon()],
                ["walbescherming", shapely.Polygon()],
            ],
            columns=["bgt-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        bgt_bak_bord_kast_paal_put_straatmeubilair = gpd.GeoDataFrame(
            [
                [np.nan, "afval apart plaats", shapely.Polygon()],
                [np.nan, "afvalbak", shapely.Polygon()],
                [np.nan, "bloembak", shapely.Polygon()],
                [np.nan, "container", shapely.Polygon()],
                [np.nan, "drinkbak", shapely.Polygon()],
                [np.nan, "zand- / zoutbak", shapely.Polygon()],
                [np.nan, "dynamische snelheidsindicator", shapely.Polygon()],
                [np.nan, "informatiebord", shapely.Polygon()],
                [np.nan, "plaatsnaambord", shapely.Polygon()],
                [np.nan, "reclamebord", shapely.Polygon()],
                [np.nan, "scheepvaartbord", shapely.Polygon()],
                [np.nan, "straatnaambord", shapely.Polygon()],
                [np.nan, "verkeersbord", shapely.Polygon()],
                [np.nan, "verklikker transportleiding", shapely.Polygon()],
                [np.nan, "waarschuwingshek", shapely.Polygon()],
                [np.nan, "wegwijzer", shapely.Polygon()],
                [np.nan, "CAI-kast", shapely.Polygon()],
                [np.nan, "elektrakast", shapely.Polygon()],
                [np.nan, "gaskast", shapely.Polygon()],
                [np.nan, "GMS kast", shapely.Polygon()],
                [np.nan, "openbare verlichtingkast", shapely.Polygon()],
                [np.nan, "rioolkast", shapely.Polygon()],
                [np.nan, "telecom kast", shapely.Polygon()],
                [np.nan, "telkast", shapely.Polygon()],
                [np.nan, "verkeersregelinstallatiekast", shapely.Polygon()],
                [np.nan, "bovenleidingmast", shapely.Polygon()],
                [np.nan, "laagspanningsmast", shapely.Polygon()],
                [np.nan, "radarmast", shapely.Polygon()],
                [np.nan, "straalzender", shapely.Polygon()],
                [np.nan, "zendmast", shapely.Polygon()],
                [np.nan, "afsluitpaal", shapely.Polygon()],
                [np.nan, "dijkpaal", shapely.Polygon()],
                [np.nan, "drukknoppaal", shapely.Polygon()],
                [np.nan, "grensmarkering", shapely.Polygon()],
                [np.nan, "haltepaal", shapely.Polygon()],
                [np.nan, "hectometerpaal", shapely.Polygon()],
                [np.nan, "lichtmast", shapely.Polygon()],
                [np.nan, "poller", shapely.Polygon()],
                [np.nan, "portaal", shapely.Polygon()],
                [np.nan, "praatpaal", shapely.Polygon()],
                [np.nan, "sirene", shapely.Polygon()],
                [np.nan, "telpaal", shapely.Polygon()],
                [np.nan, "verkeersbordpaal", shapely.Polygon()],
                [np.nan, "verkeersregelinstallatiepaal", shapely.Polygon()],
                [np.nan, "vlaggenmast", shapely.Polygon()],
                [np.nan, "benzine- / olieput", shapely.Polygon()],
                [np.nan, "brandkraan / -put", shapely.Polygon()],
                [np.nan, "drainageput", shapely.Polygon()],
                [np.nan, "gasput", shapely.Polygon()],
                [np.nan, "inspectie- / rioolput", shapely.Polygon()],
                [np.nan, "kolk", shapely.Polygon()],
                [np.nan, "waterleidingput", shapely.Polygon()],
                [np.nan, "detectielus", shapely.Polygon()],
                [np.nan, "camera", shapely.Polygon()],
                [np.nan, "debietmeter", shapely.Polygon()],
                [np.nan, "flitser", shapely.Polygon()],
                [np.nan, "GMS sensor", shapely.Polygon()],
                [np.nan, "hoogtedetectieapparaat", shapely.Polygon()],
                [np.nan, "lichtcel", shapely.Polygon()],
                [np.nan, "radar detector", shapely.Polygon()],
                [np.nan, "waterstandmeter", shapely.Polygon()],
                [np.nan, "weerstation", shapely.Polygon()],
                [np.nan, "windmeter", shapely.Polygon()],
                [np.nan, "abri", shapely.Polygon()],
                [np.nan, "bank", shapely.Polygon()],
                [np.nan, "betaalautomaat", shapely.Polygon()],
                [np.nan, "bolder", shapely.Polygon()],
                [np.nan, "brievenbus", shapely.Polygon()],
                [np.nan, "fietsenkluis", shapely.Polygon()],
                [np.nan, "fietsenrek", shapely.Polygon()],
                [np.nan, "fontein", shapely.Polygon()],
                [np.nan, "herdenkingsmonument", shapely.Polygon()],
                [np.nan, "kunstobject", shapely.Polygon()],
                [np.nan, "lichtpunt", shapely.Polygon()],
                [np.nan, "openbaar toilet", shapely.Polygon()],
                [np.nan, "parkeerbeugel", shapely.Polygon()],
                [np.nan, "picknicktafel", shapely.Polygon()],
                [np.nan, "reclamezuil", shapely.Polygon()],
                [np.nan, "slagboom", shapely.Polygon()],
                [np.nan, "speelvoorziening", shapely.Polygon()],
                [np.nan, "telefooncel", shapely.Polygon()],
                [np.nan, "waardeOnbekend", shapely.Polygon()],
                [np.nan, np.nan, shapely.Polygon()],  # this one should be deleted.
                ["niet-bgt", np.nan, shapely.Polygon()],  # this one should be deleted.
                ["waardeOnbekend", np.nan, shapely.Polygon()],  # this one should be deleted.
            ],
            columns=["function", "plus-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )

        reclassified_gdf = SmallAboveGroundObstacles._set_suitability_values(
            [bgt_scheiding_1, bgt_scheiding_2, bgt_bak_bord_kast_paal_put_straatmeubilair], weight_values
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                    85,
                ]
            ),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )

        assert "niet-bgt" not in reclassified_gdf["bgt-type"]
        assert "waardeOnbekend" not in reclassified_gdf["plus-type"]

    def test_vegetation_object(self):
        weight_values = {
            # plus_type
            "haag": 1,
            "boom": 2,
            "waardeOnbekend": 1,
        }
        gdf_1 = gpd.GeoDataFrame(
            [
                ["haag", shapely.LineString([[0, 0], [1, 0], [1, 1]])],
                ["haag", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
                ["waardeOnbekend", shapely.Polygon()],
            ],
            columns=["plus-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        gdf_2 = gpd.GeoDataFrame(
            [
                ["boom", shapely.Point(1, 1)],
            ],
            columns=["plus-type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = VegetationObject._set_suitability_values([gdf_1, gdf_2], weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([1, 1, 2]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 1, 2]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        assert "waardeOnbekend" not in reclassified_gdf["plus-type"]

        reclassified_gdf = VegetationObject._update_geometry_values(reclassified_gdf, {"boom": 5})
        assert reclassified_gdf.is_empty.value_counts().get(False) == 3
        assert reclassified_gdf.is_empty.value_counts().get(True) is None
        assert reclassified_gdf.geom_type.unique().tolist() == ["LineString", "Polygon"]

    def test_wegdeel(self):
        weight_values = {
            "baan voor vliegverkeer": 1,
            "fietspad": 2,
            "inrit": 3,
            "OV-baan": 4,
            "overweg": 5,
            "parkeervlak": 6,
            "rijbaan autosnelweg": 7,
            "rijbaan autoweg": 8,
            "rijbaan lokale weg": 9,
            "rijbaan regionale weg": 10,
            "ruiterpad": 11,
            "spoorbaan": 12,
            "voetgangersgebied": 13,
            "voetpad": 14,
            "voetpad op trap": 15,
            "woonerf": 16,
            "gesloten verharding": 17,
            "half verhard": 18,
            "onverhard": 19,
            "open verharding": 20,
        }
        gdf_1 = gpd.GeoDataFrame(
            [
                ["baan voor vliegverkeer", "gesloten verharding", shapely.Polygon()],
                ["fietspad", "gesloten verharding", shapely.Polygon()],
                ["inrit", "gesloten verharding", shapely.Polygon()],
                ["OV-baan", "gesloten verharding", shapely.Polygon()],
                ["overweg", "gesloten verharding", shapely.Polygon()],
                ["parkeervlak", "gesloten verharding", shapely.Polygon()],
                ["rijbaan autosnelweg", "gesloten verharding", shapely.Polygon()],
                ["rijbaan autoweg", "gesloten verharding", shapely.Polygon()],
                ["rijbaan lokale weg", "gesloten verharding", shapely.Polygon()],
                ["rijbaan regionale weg", "gesloten verharding", shapely.Polygon()],
                ["ruiterpad", "gesloten verharding", shapely.Polygon()],
                ["spoorbaan", "gesloten verharding", shapely.Polygon()],
                ["voetgangersgebied", "gesloten verharding", shapely.Polygon()],
                ["voetpad", "gesloten verharding", shapely.Polygon()],
                ["voetpad op trap", "half verhard", shapely.Polygon()],
                ["woonerf", "onverhard", shapely.Polygon()],
            ],
            columns=["function", "surfaceMaterial", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = Wegdeel._set_suitability_values(gdf_1, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_1"],
            pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["sv_2"],
            pd.Series(
                [
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    17,
                    18,
                    19,
                ]
            ),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )

    def test_excluded_area(self):
        weight_values = {
            "constraint": 1,
        }
        gdf_1 = gpd.GeoDataFrame(
            [
                ["the mayors back garden", shapely.Polygon([[1, 1], [0, 0], [1, 0], [1, 1]])],
                ["a very nice park", shapely.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])],
            ],
            columns=["description", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = ExcludedArea._set_suitability_values(gdf_1, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"],
            pd.Series([1, 1]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        assert reclassified_gdf.is_empty.value_counts().get(False) == 2
        assert reclassified_gdf.is_empty.value_counts().get(True) is None
        assert reclassified_gdf.geom_type.unique().tolist() == ["Polygon"]

    def test_existing_utilities(self):
        weight_values = {
            "hoogspanning_bovengronds": 1,
            "hoogspanning_ondergronds": 2,
            "gasunie_leidingen": 3,
            "alliander_stationsterrein": 4,
        }
        buffer_values = {
            "hoogspanning_bovengronds_buffer": 5,
            "hoogspanning_ondergronds_buffer": 6,
            "gasunie_leidingen_buffer": 7,
        }
        gdf_high_voltage_above_ground = gpd.GeoDataFrame(
            [
                [1, 150, "high_voltage_cable_overhead", shapely.LineString([[3, 3], [1, 0]])],
                [1, 0, "high_voltage_cable_overhead", shapely.LineString([[2, 2], [1, 0]])],
            ],
            columns=["OBJECTID", "SPANNINGSNIVEAU", "type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        gdf_high_voltage_under_ground = gpd.GeoDataFrame(
            [
                [1, 50, "high_voltage_cable_underground", shapely.LineString([[0, 0], [1, 0]])],
                [1, 0, "high_voltage_cable_underground", shapely.LineString([[0, 0], [1, 0]])],
            ],
            columns=["OBJECTID", "SPANNINGSNIVEAU", "type", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        gasunie_leiding_gdf = gpd.GeoDataFrame(
            [
                [1, "123", "In Bedrijf", shapely.LineString([[0, 0], [1, 0]])],
                [2, "1234", "Niet in Bedrijf", shapely.LineString([[0, 0], [1, 0]])],
            ],
            columns=["OBJECTID", "Leiding", "StatusOperationeel", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        gdf_substation_terrain = gpd.GeoDataFrame(
            [
                [1, "1 000 001", shapely.Point(0, 0).buffer(200)],  # high voltage substation, should be kept.
                [2, "1 000 002", shapely.Point(0, 0).buffer(1)],  # regular substation, should be deleted.
            ],
            columns=["OBJECTID", "STATIONCOMPLEX", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = ExistingUtilities._set_suitability_and_geometry_values(
            [gdf_high_voltage_above_ground, gdf_high_voltage_under_ground, gasunie_leiding_gdf, gdf_substation_terrain],
            weight_values,
            buffer_values,
        )

        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"].sort_values(),
            pd.Series([1, 2, 3, 4]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        assert reclassified_gdf.geom_type.unique().tolist() == ["Polygon"]
        assert "Niet in Bedrijf" not in reclassified_gdf["StatusOperationeel"]
        assert "1 000 002" not in reclassified_gdf["STATIONCOMPLEX"]

    def test_existing_substations(self):
        weight_values = {
            "alliander_middenspanningsstation": 1,
        }
        gdf_substations = gpd.GeoDataFrame(
            [
                [1, "1 000 001", shapely.Point(0, 0).buffer(5)],
                [2, "1 000 002", shapely.Point(0, 0).buffer(5)],
            ],
            columns=["OBJECTID", "STATIONCOMPLEX", "geometry"],
            crs=Config.CRS,
            geometry="geometry",
        )
        reclassified_gdf = ExistingSubstations._set_suitability_values(gdf_substations, weight_values)
        pd.testing.assert_series_equal(
            reclassified_gdf["suitability_value"].sort_values(),
            pd.Series([1, 1]),
            check_names=False,
            check_exact=True,
            check_dtype=False,
            check_index=False,
        )
        assert reclassified_gdf.geom_type.unique().tolist() == ["Polygon"]
