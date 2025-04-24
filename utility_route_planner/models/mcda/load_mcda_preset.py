# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
import typing

import pydantic
import shapely
import structlog
from pydantic import model_validator, ConfigDict, field_validator

from utility_route_planner.models.mcda.exceptions import InvalidGroupValue, InvalidSuitabilityValue, InvalidLayerName
from utility_route_planner.models.mcda.mcda_presets import preset_collection
from settings import Config

from utility_route_planner.models.mcda.vector_preprocessing.base import VectorPreprocessorBase

logger = structlog.get_logger(__name__)


class RasterPresetCriteria(pydantic.BaseModel):
    """
    Class for defining the datamodel for a criteria in the raster preset. Is part of the RasterPreset datamodel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    description: str = pydantic.Field(description="Description of the criteria.")
    layer_names: list = pydantic.Field(description="Layer names in the geopackage which will be handled.")
    preprocessing_function: VectorPreprocessorBase
    group: str = pydantic.Field(description="Determines how the criteria is handled.")
    weight_values: dict = pydantic.Field(..., description="Contains values for defining how important the layer is.")
    geometry_values: typing.Optional[dict] = pydantic.Field(
        default=None,
        description="Contains values for optional computational geometry steps, e.g., buffer.",
    )

    @model_validator(mode="after")
    def validate_attributes(self):
        group = self.group
        weight_values = self.weight_values

        self.validate_group(group)
        self.validate_weights(weight_values)

        return self

    @staticmethod
    def validate_weights(weight_values):
        for name, weight in weight_values.items():
            if not type(weight) == int:  # noqa: E721
                raise InvalidSuitabilityValue(
                    f"{name} has an invalid value of {weight}. Expected int, received {type(weight)}"
                )
            if (
                not Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER
                <= weight
                <= Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER
            ):
                raise InvalidSuitabilityValue(
                    f"{name} has an invalid value of {weight}. Weights must be between: {Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER}-{Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER}"
                )

    @staticmethod
    def validate_group(group):
        if group not in ["a", "b", "c"]:
            raise InvalidGroupValue(f"Group must be 'a', 'b' or 'c'. Received: {group}")


class RasterPresetGeneral(pydantic.BaseModel):
    """
    Check if we have the necessary general settings for the cost-surface generation using MCDA.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    description: typing.Optional[str] = pydantic.Field(..., description="Description of the preset.")
    prefix: str = pydantic.Field(
        ...,
        description="Prefix to apply for all files relevant for a given preset.",
    )
    final_raster_name: str = pydantic.Field(default="zz_test_raster", description="Name of the final raster.")
    project_area_geometry: shapely.MultiPolygon | shapely.Polygon = pydantic.Field(
        ...,
        description="Shapely geometry to use defining the project area for the utility route.",
    )
    path_input_geopackage: pathlib.Path = pydantic.Field(
        ..., description="Path to the input geopackage containing all input data for MCDA."
    )

    @field_validator("project_area_geometry")
    def validate_group(cls, v: shapely.MultiPolygon | shapely.Polygon) -> shapely.MultiPolygon:
        if v.geom_type == "Polygon":
            v = shapely.MultiPolygon([v])
        if shapely.get_num_geometries(v) < 1:
            raise ValueError("Input project MultiPolygon is not valid as it does not contain 1 or more geometries.")
        return v


class RasterPreset(pydantic.BaseModel):
    """
    Class for defining the datamodel of a raster preset. A raster preset contains:
    - General settings such as, but not limited to: names, directories, raster settings.
    - List of criteria to include in the raster.
    """

    general: RasterPresetGeneral
    criteria: typing.Dict[str, RasterPresetCriteria]


def validate_layer_names(existing_layers, layer_names):
    for layer_name in layer_names:
        if not isinstance(layer_name, str):
            raise InvalidLayerName(f"Invalid value in layer_names: {layer_name}. Should be a string.")
        if layer_name not in existing_layers:
            raise InvalidLayerName(f"{layer_name} is not in the source geopackage. Existing layers: {existing_layers}")


def load_preset(preset_name: str | dict, path_input_geopackage, project_area_geometry: shapely.Polygon) -> RasterPreset:
    """
    Convert the raw configuration file to a pydantic datamodel.

    :param project_area_geometry: project area in which the route must be calculated.
    :param preset_name: preset dictionary to load from the mcda_presets.py.
    :param path_input_geopackage: path to the geopackage containing the vector geodata to process in a cost-surface.
    :return: datamodel containing configuration for the raster to create.
    """

    try:
        if isinstance(preset_name, str):
            preset_to_load_raw = preset_collection.get(preset_name)
            if preset_to_load_raw is None:
                logger.error(
                    f"Preset: {preset_to_load_raw} is not implemented. Options are: {preset_collection.keys()}"
                )
                raise ValueError
        elif isinstance(preset_name, dict):
            preset_to_load_raw = preset_name
        else:
            logger.error(f"Unsupported input received. Expecting dict or str, received {type(preset_name)}.")
            raise ValueError
        # Add the geopackage to use to the raw dictionary to convert to a pydantic model.
        preset_to_load_raw["general"]["path_input_geopackage"] = path_input_geopackage
        preset_to_load_raw["general"]["project_area_geometry"] = project_area_geometry
        preset_model = RasterPreset(
            general=preset_to_load_raw["general"],
            criteria=preset_to_load_raw["criteria"],
        )

        logger.info("Successfully loaded the raster preset datamodel.")
        return preset_model

    except pydantic.ValidationError as e:
        logger.error(f"Exception as str: {e}")
        logger.error(e)
        raise


if __name__ == "__main__":
    # Helper function for creating an Excel overview of all available criteria
    import pandas as pd
    import geopandas as gpd

    benchmark_preset = load_preset(
        Config.RASTER_PRESET_NAME_BENCHMARK,
        Config.PYTEST_PATH_GEOPACKAGE_MCDA,
        gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA).iloc[0].geometry,
    )
    data = []
    for criterion, details in benchmark_preset.criteria.items():
        for weight_name, weight_value in details.weight_values.items():
            data.append((criterion, weight_name, weight_value, details.group))

    df = pd.DataFrame(data, columns=["Criteria", "weight_name", "Weight Value", "group"])
    df.to_csv(Config.PATH_RESULTS / "mcda_weights.csv")
