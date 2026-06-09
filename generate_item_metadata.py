import logging
import os
import json
import pystac
import xarray as xr
from datetime import datetime
from pathlib import Path
from shapely.geometry import box, mapping
from typing import List, Dict, Any, Optional, Tuple


from config import (
    APP_LOGGER_NAME,
    IS_OVERWRITE_S3,
    IS_UPLOAD_S3,
    ITEM_CONFIG_FILE_NAME,
    ITEM_CONFIG_OPTIONAL,
    ITEM_FOLDER_LEVEL,
    ITEM_FOLDER_LEVEL_DD,
    ITEM_FOLDER_LEVEL_MM,
    ITEM_FOLDER_LEVEL_NONE,
    ITEM_FOLDER_LEVEL_YYYY,
    ADDITIONAL_PROPERTY_KEYS,
    S3_ENDPOINT_URL,
    S3_USER_GENERATED_BUCKET_PREFIX,
    STAC_VERSION,
)

import usergenerated.logging_config  # import must come before other modules in this project so that logging setup correctly
from usergenerated import datetools
from usergenerated.config import confighelper
from usergenerated.item import itemhelper
from usergenerated.s3tools import S3Tools
from usergenerated.env_utils import validate_aws_credentials


def infer_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    for lat_name, lon_name in (("lat", "lon"), ("latitude", "longitude")):
        if lat_name in ds.coords and lon_name in ds.coords:
            return lat_name, lon_name
        if lat_name in ds and lon_name in ds:
            return lat_name, lon_name
    raise KeyError("Could not find lat/lon coords (tried lat/lon and latitude/longitude).")


def _normalize_attrs(attrs: Dict[str, Any]) -> Dict[str, str]:
    return {str(k).lower(): str(v) for k, v in attrs.items()}


def _first_nonempty(attrs: Dict[str, str], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = attrs.get(key.lower(), "")
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def read_item_nc_metadata(item_folder_path: Path) -> Tuple[Optional[List[float]], Dict[str, str]]:
    """
    Read bbox and selected metadata from one representative NetCDF file in the item folder.

    Returned metadata keys:
      - model
      - experiment
      - generation
      - realization
      - resolution
      - activity

    If no NetCDF file is found, returns (None, {}).
    """
    nc_files = sorted(item_folder_path.rglob("*.nc"))
    if not nc_files:
        return None, {}

    fp = nc_files[0]
    ds = xr.open_dataset(fp)
    try:
        lat_name, lon_name = infer_lat_lon_names(ds)
        lat = ds[lat_name]
        lon = ds[lon_name]
        lat_min, lat_max = float(lat.min()), float(lat.max())
        lon_min, lon_max = float(lon.min()), float(lon.max())
        bbox = [lon_min, lat_min, lon_max, lat_max]

        attrs = _normalize_attrs(ds.attrs)

        meta = {
            "model": _first_nonempty(
                attrs,
                ["model", "driving_model", "source_id", "gcm", "institute_model"],
                "",
            ),
            "experiment": _first_nonempty(
                attrs,
                ["experiment", "experiment_id", "scenario"],
                "",
            ),
            "generation": _first_nonempty(
                attrs,
                ["generation", "product_version", "version"],
                "",
            ),
            "realization": _first_nonempty(
                attrs,
                ["realization", "member_id", "variant_label", "ensemble_member"],
                "",
            ),
            "resolution": _first_nonempty(
                attrs,
                ["resolution", "spatial_resolution", "grid_resolution"],
                "",
            ),
            "activity": _first_nonempty(
                attrs,
                ["activity", "activity_id", "project"],
                "",
            ),
        }

        return bbox, meta
    finally:
        ds.close()


def infer_data_type_from_item_id(item_id: str) -> str:
    item_id_lower = item_id.lower()
    if "__monthly" in item_id_lower:
        return "monthly"
    if "__yearly" in item_id_lower:
        return "yearly"
    return ""


class ItemGenerator:

    def __init__(self, collection_id: str, overide_bucket_name: Optional[str] = None) -> None:
        """
        Class to generate STAC Item metadata for a given collection.

        This generator supports a mixed collection where:
        - regular monthly items are stored under data/YYYY/MM/<item_id>
        - yearly special-case items are also stored under MM layout, usually in
          data/YYYY/01/<item_id>, while the item_id itself carries a full-year
          time span (YYYY0101 -> YYYY1231)

        This version injects explicit STAC properties by reading one
        representative NetCDF file from each item folder:
        - model
        - experiment
        - generation
        - realization
        - resolution
        - activity
        - data_type

        Item ids are expected to encode:
        - model
        - experiment
        - generation
        - realization
        - resolution
        - activity
        - data_type
        """
        self.collection_id = collection_id

        if "-" in collection_id:
            raise ValueError(
                "collection_id cannot contain a dash ('-'). Only underscores ('_') are permitted in a dedl Collection ID."
            )

        self.overide_bucket_name = overide_bucket_name
        self.collection_root = Path(collection_id)
        self.collection_path = Path(f"{collection_id}/metadata/collection.json")
        self.collection_config_path = Path(f"{collection_id}/metadata/collection_config.json")
        self.items_root = Path(f"{collection_id}/metadata/items")
        self.data_root = Path(f"{collection_id}/data")

        self.is_overwrite_s3 = IS_OVERWRITE_S3
        self.aws_access_key_id, self.aws_secret_access_key = validate_aws_credentials()

        self.s3tools = S3Tools(
            self.aws_access_key_id, self.aws_secret_access_key, self.is_overwrite_s3
        )

    def _get_item_folders_by_level(self, folder_level: str) -> List[Path]:
        item_folder_paths: List[Path] = []

        if folder_level == ITEM_FOLDER_LEVEL_DD:
            for year_folder in self.data_root.iterdir():
                if year_folder.is_dir():
                    for month_folder in year_folder.iterdir():
                        if month_folder.is_dir():
                            for day_folder in month_folder.iterdir():
                                if day_folder.is_dir():
                                    for item_folder in day_folder.iterdir():
                                        if item_folder.is_dir():
                                            item_folder_paths.append(item_folder)

        elif folder_level == ITEM_FOLDER_LEVEL_MM:
            for year_folder in self.data_root.iterdir():
                if year_folder.is_dir():
                    for month_folder in year_folder.iterdir():
                        if month_folder.is_dir():
                            for item_folder in month_folder.iterdir():
                                if item_folder.is_dir():
                                    item_folder_paths.append(item_folder)

        elif folder_level == ITEM_FOLDER_LEVEL_YYYY:
            for year_folder in self.data_root.iterdir():
                if year_folder.is_dir():
                    for item_folder in year_folder.iterdir():
                        if item_folder.is_dir():
                            item_folder_paths.append(item_folder)

        elif folder_level == ITEM_FOLDER_LEVEL_NONE:
            for item_folder in self.data_root.iterdir():
                if item_folder.is_dir():
                    item_folder_paths.append(item_folder)

        else:
            raise ValueError(
                f"Unexpected ITEM_FOLDER_LEVEL configuration: '{folder_level}'. "
                f"Expected values: {ITEM_FOLDER_LEVEL_YYYY}, {ITEM_FOLDER_LEVEL_MM}, "
                f"{ITEM_FOLDER_LEVEL_DD}, or {ITEM_FOLDER_LEVEL_NONE}"
            )

        return item_folder_paths

    def run(self) -> None:
        collection = confighelper.load_and_validate_collection(
            self.collection_path,
            self.collection_id,
            save_reordered_collection=False,
            is_compare_expected_id=True,
        )

        collection_config = confighelper.load_config(self.collection_config_path)
        logger.debug(f"collection_config:{collection_config}")

        if ITEM_FOLDER_LEVEL not in collection_config:
            collection_config[ITEM_FOLDER_LEVEL] = ITEM_FOLDER_LEVEL_DD

        item_folder_paths = self._get_item_folders_by_level(collection_config[ITEM_FOLDER_LEVEL])
        self.is_simplified_process = collection_config[ITEM_FOLDER_LEVEL] == ITEM_FOLDER_LEVEL_NONE

        for item_folder_path in item_folder_paths:
            if not self.is_simplified_process:
                self.create_item(item_folder_path, collection, collection_config)
            else:
                self.create_item_simplified_process(item_folder_path, collection, collection_config)

        confighelper.sort_item_assets_in_folder(self.items_root)

        if IS_UPLOAD_S3:
            bucket_name = f"{S3_USER_GENERATED_BUCKET_PREFIX}-{self.collection_id.lower()}"
            if self.overide_bucket_name:
                bucket_name = self.overide_bucket_name

            self.s3tools.upload_folder_to_s3(
                str(self.collection_root),
                S3_ENDPOINT_URL,
                bucket_name,
            )

    def get_item(
        self,
        item_id: str,
        geometry: Optional[Dict[str, Any]],
        bbox: Optional[List[float]],
        item_datetime: datetime,
        item_properties: Dict[str, Any],
        item_folder_path: Path,
        config_list: List[Dict[str, Any]],
        collection: pystac.Collection,
    ) -> None:
        pystac.set_stac_version(STAC_VERSION)

        item = pystac.Item(
            id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=item_datetime,
            properties=item_properties,
        )

        asset_paths = [file for file in item_folder_path.rglob("*") if file.is_file()]

        for asset_path in asset_paths:
            asset_href_data_root = Path(*asset_path.parts[1:])
            asset_href = asset_path.relative_to(item_folder_path)
            logger.info(f"asset_href:{asset_href}")

            item_asset_ignore_list = confighelper.get_config_value(config_list, "item_asset_ignore_list")
            if asset_href.name not in item_asset_ignore_list:
                media_type = itemhelper.get_media_type(asset_path)
                thumbnail_regex = confighelper.get_config_value(config_list, "thumbnail_regex")
                overview_regex = confighelper.get_config_value(config_list, "overview_regex")

                item.add_asset(
                    key=str(asset_href),
                    asset=pystac.Asset(
                        href=str(asset_href_data_root),
                        media_type=media_type,
                        roles=itemhelper.get_asset_role(
                            media_type, asset_href, thumbnail_regex, overview_regex
                        ),
                    ),
                )

        collection.set_self_href(
            f"https://hda.data.destination-earth.eu/stac/v2/collections/{self.collection_id}"
        )
        item.set_collection(collection)
        item.set_self_href(
            f"https://hda.data.destination-earth.eu/stac/v2/collections/{self.collection_id}/items/{item_id}"
        )

        self.items_root.mkdir(parents=True, exist_ok=True)
        item_path = self.items_root / f"{item.id}.json"
        with open(item_path, "w") as f:
            json.dump(item.to_dict(), f, indent=4)

        logger.info(f"Item metadata saved to {item_path}\n")

    def _merge_explicit_nc_properties(
        self,
        item_folder_path: Path,
        item_id: str,
        item_properties: Dict[str, Any],
        config_list: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
        """
        Add explicit STAC properties from one representative NetCDF file.

        Rules:
        - model, experiment, generation, realization, resolution, activity
          are read from the NetCDF global attributes when present
        - data_type is inferred from item_id if present as __monthly / __yearly
        - bbox comes from config if available, otherwise from the NetCDF file
        """
        nc_bbox, nc_meta = read_item_nc_metadata(item_folder_path)
        additional_keys = confighelper.get_config_value(config_list, ADDITIONAL_PROPERTY_KEYS, True) or []

        for key in additional_keys:
            if key in nc_meta and nc_meta[key]:
                item_properties[key] = nc_meta[key]

        if "data_type" in additional_keys:
            inferred_data_type = infer_data_type_from_item_id(item_id)
            if inferred_data_type:
                item_properties["data_type"] = inferred_data_type

        bbox = confighelper.get_config_value(config_list, "bbox", True)
        if not bbox and nc_bbox:
            bbox = nc_bbox

        if bbox:
            polygon = box(*bbox)
            geometry = mapping(polygon)
        else:
            geometry = None

        return geometry, bbox

    def create_item_simplified_process(self, item_folder_path: Path, collection: pystac.Collection, collection_config: dict):
        try:
            item_config = confighelper.load_config(
                item_folder_path / ITEM_CONFIG_FILE_NAME,
                is_config_file_optional=True,
            )
            logger.debug(item_config)
            config_list = [item_config, collection_config]

            item_id = item_folder_path.name
            item_folder_path_parts = str(item_folder_path).split(os.path.sep)
            collection_id = item_folder_path_parts[0]

            if not item_id.startswith(collection_id):
                item_id = collection_id + "_" + item_id

            year_from_folder_path = item_folder_path_parts[2]
            item_date_overide = itemhelper.get_item_date_overide(config_list)
            if item_date_overide:
                datetime_from_folder_path = item_date_overide
            elif itemhelper.is_valid_year(year_from_folder_path):
                datetime_from_folder_path = itemhelper.get_datetime_from_folder_path(
                    item_folder_path_parts, collection_config, item_id, collection_id
                )
            else:
                datetime_from_folder_path = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

            item_datetime = datetime_from_folder_path
            item_properties = {}
            item_properties.update(confighelper.get_config_value(config_list, "properties", True) or {})
            datetools.is_same_day(datetime_from_folder_path, item_datetime)

            geometry, bbox = self._merge_explicit_nc_properties(
                item_folder_path,
                item_id,
                item_properties,
                config_list,
            )

            item = self.get_item(
                item_id,
                geometry,
                bbox,
                item_datetime,
                item_properties,
                item_folder_path,
                config_list,
                collection,
            )
            return item

        except Exception as e:
            logger.error(f"Error creating item: {e}")
            return None

    def create_item(self, item_folder_path: Path, collection: pystac.Collection, collection_config: dict):
        try:
            is_item_config_optional = True
            if ITEM_CONFIG_OPTIONAL in collection_config:
                is_item_config_optional = collection_config[ITEM_CONFIG_OPTIONAL]
                print(f"{ITEM_CONFIG_OPTIONAL}:{is_item_config_optional}")

            item_config = confighelper.load_config(
                item_folder_path / ITEM_CONFIG_FILE_NAME,
                is_config_file_optional=is_item_config_optional,
            )
            logger.debug(item_config)
            config_list = [item_config, collection_config]

            logger.info(f"item_folder_path:{item_folder_path}")
            item_id = item_folder_path.name
            item_folder_path_parts = str(item_folder_path).split(os.path.sep)
            collection_id = item_folder_path_parts[0]

            datetime_from_folder_path = itemhelper.get_datetime_from_folder_path(
                item_folder_path_parts, collection_config, item_id, collection_id
            )

            item_id_property_keys = [
                "model",
                "experiment",
                "generation",
                "realization",
                "resolution",
                "activity",
                "data_type",
            ]

            (item_datetime, item_properties) = itemhelper.get_item_properties(
                item_id,
                collection_id,
                item_id_property_keys,
            )

            item_properties.update(confighelper.get_config_value(config_list, "properties", True) or {})
            datetools.is_same_day(datetime_from_folder_path, item_datetime)

            geometry, bbox = self._merge_explicit_nc_properties(
                item_folder_path,
                item_id,
                item_properties,
                config_list,
            )

            item = self.get_item(
                item_id,
                geometry,
                bbox,
                item_datetime,
                item_properties,
                item_folder_path,
                config_list,
                collection,
            )
            return item

        except Exception as e:
            logger.error(f"Error creating item: {e}")
            return None


if __name__ == "__main__":
    logger = logging.getLogger(APP_LOGGER_NAME)
    collection_id = "EO.FMI.DAT.DESTINE_CLIMATE_WILDFIRE_FWI"
    item_generator = ItemGenerator(collection_id)
    item_generator.run()
