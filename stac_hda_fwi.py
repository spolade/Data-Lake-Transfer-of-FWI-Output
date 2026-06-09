#!/usr/bin/env python3
"""
FWI STAC staging script
- Monthly STAC items for daily FWI files
- Yearly STAC items for annual statistics and yearly regional daily climatology files
- Hard-link staging where possible
- No item_config.json
- bbox written once in metadata/collection_config.json
- item_folder_level kept as MM

Final version:
- item ids include:
  model, experiment, generation, realization, resolution, activity, data_type
- yearly item metadata is derived from representative DAILY files
- no sim stamp in item ids
"""

from __future__ import annotations

import calendar
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pystac
import xarray as xr
import yaml


# ---------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------

DAILY_FWI_PATTERN = re.compile(
    r"^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_T(?P<hour>\d{2})_(?P<minute>\d{2})_fwi\.nc$",
    re.IGNORECASE,
)

ANNUAL_BASIC_PATTERN = re.compile(
    r"^(?P<year>\d{4})_fwi_basic_and_days\.nc$",
    re.IGNORECASE,
)

ANNUAL_QUANT_PATTERN = re.compile(
    r"^(?P<year>\d{4})_fwi_quantiles\.nc$",
    re.IGNORECASE,
)

ANNUAL_DAILYCLIM_PATTERN = re.compile(
    r"^(?P<year>\d{4})_fwi_dailyclim_(?P<domain>[A-Za-z]+)\.nc$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def experiment_from_year(year: int) -> str:
    if 1990 <= year <= 2014:
        return "historical"
    if 2015 <= year <= 2049:
        return "ssp3-7.0"
    raise ValueError(f"Year {year} outside expected ranges (1990–2049).")


def infer_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    for lat_name, lon_name in (("lat", "lon"), ("latitude", "longitude")):
        if lat_name in ds.coords and lon_name in ds.coords:
            return lat_name, lon_name
        if lat_name in ds and lon_name in ds:
            return lat_name, lon_name
    raise KeyError("Could not find lat/lon coords (tried lat/lon and latitude/longitude).")


def normalize_attr_dict(attrs: dict) -> Dict[str, str]:
    return {str(k).lower(): str(v) for k, v in attrs.items()}


def first_nonempty(attrs: Dict[str, str], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = attrs.get(key.lower(), "")
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def sanitize_item_token(value: str, default: str = "") -> str:
    """
    Make a metadata value safe for use in STAC item ids / folder names.
    """
    v = (value or "").strip()
    if not v:
        return default

    v = v.replace(" ", "-").replace("/", "-").replace("\\", "-")
    return v


def normalize_experiment(value: str, year: int) -> str:
    v = (value or "").strip().lower()
    if v in ["hist", "historical"]:
        return "historical"
    if v in ["ssp370", "ssp3-7.0", "ssp3_7.0", "ssp3-70"]:
        return "ssp3-7.0"
    return experiment_from_year(year)


def read_nc_metadata(fp: Path) -> Tuple[List[float], Dict[str, str]]:
    """
    Open one representative NetCDF file and read:
    - bbox from lat/lon
    - selected global attributes
    """
    ds = xr.open_dataset(fp)
    try:
        lat_name, lon_name = infer_lat_lon_names(ds)
        lat = ds[lat_name]
        lon = ds[lon_name]
        lat_min, lat_max = float(lat.min()), float(lat.max())
        lon_min, lon_max = float(lon.min()), float(lon.max())
        bbox = [lon_min, lat_min, lon_max, lat_max]

        attrs = normalize_attr_dict(ds.attrs)

        meta = {
            "model": first_nonempty(
                attrs,
                ["model", "driving_model", "source_id", "gcm", "institute_model"],
                "",
            ),
            "experiment": first_nonempty(
                attrs,
                ["experiment", "experiment_id", "scenario"],
                "",
            ),
            "generation": first_nonempty(
                attrs,
                ["generation", "product_version", "version"],
                "",
            ),
            "realization": first_nonempty(
                attrs,
                ["realization", "member_id", "variant_label", "ensemble_member"],
                "",
            ),
            "resolution": first_nonempty(
                attrs,
                ["resolution", "spatial_resolution", "grid_resolution"],
                "",
            ),
            "activity": first_nonempty(
                attrs,
                ["activity", "activity_id", "project"],
                "",
            ),
        }
        return bbox, meta
    finally:
        ds.close()


def require_metadata_fields(meta: Dict[str, str], required_keys: List[str], context: str = "") -> Dict[str, str]:
    missing = [k for k in required_keys if not str(meta.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Missing required metadata {missing}. Context: {context}")
    return meta


def stage_file_with_hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def get_annual_stats_dir(model_dir: Path) -> Path:
    return model_dir / "Annual_Stat"


def create_collection_config(metadata_dir: Path, eumet_id: str, bbox: List[float]) -> None:
    cfg = {
        "id": eumet_id,
        "item_asset_ignore_list": ["item_config.json"],
        "item_config_optional": True,
        "item_folder_level": "MM",
        "YYYY": "MM",
        "thumbnail_regex": "^thumbnail",
        "overview_regex": "^overview",
        "additional_property_keys": [
            "model",
            "experiment",
            "generation",
            "realization",
            "resolution",
            "activity",
            "data_type",
        ],
        "bbox": bbox,
    }
    with open(metadata_dir / "collection_config.json", "w") as f:
        json.dump(cfg, f, indent=4)


def create_stac_collection(
    collection_path: Path,
    spatial_extent: List[float],
    temporal_extent: Tuple[datetime, datetime],
    eumet_id: str,
    title: str,
    description: str,
    short_description: str | None,
) -> None:
    collection = pystac.Collection(
        id=eumet_id,
        description=description,
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([spatial_extent]),
            temporal=pystac.TemporalExtent([[temporal_extent[0], temporal_extent[1]]]),
        ),
        license="CC-BY-4.0",
        title=title,
        keywords=["NetCDF", "Climate", "Wildfire", "FWI"],
        providers=[
            pystac.Provider(
                name="Finnish Meteorological Institute (FMI)",
                roles=["producer", "licensor"],
                url="https://en.ilmatieteenlaitos.fi",
            )
        ],
        extra_fields=(
            {"dedl:short_description": short_description}
            if short_description
            else None
        ),
    )
    collection.save_object(dest_href=str(collection_path))


def build_monthly_item_id(
    eumet_id: str,
    year: int,
    month: int,
    model: str,
    experiment: str,
    generation: str,
    realization: str,
    resolution: str,
    activity: str,
    data_type: str,
) -> str:
    last_day = calendar.monthrange(year, month)[1]

    safe_model = sanitize_item_token(model)
    safe_experiment = sanitize_item_token(experiment)
    safe_generation = sanitize_item_token(generation)
    safe_realization = sanitize_item_token(realization)
    safe_resolution = sanitize_item_token(resolution)
    safe_activity = sanitize_item_token(activity)
    safe_data_type = sanitize_item_token(data_type)

    return (
        f"{eumet_id}_"
        f"{year}{month:02d}01T000000_"
        f"{year}{month:02d}{last_day:02d}T230000__"
        f"{safe_model}__{safe_experiment}__{safe_generation}__"
        f"{safe_realization}__{safe_resolution}__{safe_activity}__"
        f"{safe_data_type}"
    )


def build_yearly_item_id(
    eumet_id: str,
    year: int,
    model: str,
    experiment: str,
    generation: str,
    realization: str,
    resolution: str,
    activity: str,
    data_type: str,
) -> str:
    safe_model = sanitize_item_token(model)
    safe_experiment = sanitize_item_token(experiment)
    safe_generation = sanitize_item_token(generation)
    safe_realization = sanitize_item_token(realization)
    safe_resolution = sanitize_item_token(resolution)
    safe_activity = sanitize_item_token(activity)
    safe_data_type = sanitize_item_token(data_type)

    return (
        f"{eumet_id}_"
        f"{year}0101T000000_"
        f"{year}1231T230000__"
        f"{safe_model}__{safe_experiment}__{safe_generation}__"
        f"{safe_realization}__{safe_resolution}__{safe_activity}__"
        f"{safe_data_type}"
    )


# ---------------------------------------------------------------------
# File iterators
# ---------------------------------------------------------------------

def iter_daily_model_files(appdata_dir: Path) -> Iterable[Tuple[str, Path]]:
    """
    Yield daily FWI files under model directories.
    Fallback model name comes from directory name if missing in file attrs.
    """
    for model_dir in sorted([p for p in appdata_dir.iterdir() if p.is_dir()]):
        model_from_dir = model_dir.name.strip()

        for member_dir in sorted(model_dir.glob("*/member01")):
            if not member_dir.is_dir():
                continue

            for fp in sorted(member_dir.iterdir()):
                if not fp.is_file():
                    continue
                if fp.suffix.lower() != ".nc":
                    continue
                if not DAILY_FWI_PATTERN.match(fp.name):
                    continue

                yield model_from_dir, fp


def iter_annual_model_files(appdata_dir: Path) -> Iterable[Tuple[str, int, List[Path]]]:
    """
    For each model, collect annual files by year from that model's Annual_Stat directory.
    """
    for model_dir in sorted([p for p in appdata_dir.iterdir() if p.is_dir()]):
        model_from_dir = model_dir.name.strip()
        annual_stats_dir = get_annual_stats_dir(model_dir)

        if not annual_stats_dir.exists():
            print(f"[WARN] Annual_Stat directory not found for model {model_from_dir}: {annual_stats_dir}")
            continue

        annual_groups: Dict[int, List[Path]] = defaultdict(list)

        for fp in sorted(annual_stats_dir.iterdir()):
            if not fp.is_file():
                continue
            if fp.suffix.lower() != ".nc":
                continue

            m1 = ANNUAL_BASIC_PATTERN.match(fp.name)
            m2 = ANNUAL_QUANT_PATTERN.match(fp.name)
            m3 = ANNUAL_DAILYCLIM_PATTERN.match(fp.name)

            if m1:
                year = int(m1.group("year"))
                annual_groups[year].append(fp)
            elif m2:
                year = int(m2.group("year"))
                annual_groups[year].append(fp)
            elif m3:
                year = int(m3.group("year"))
                annual_groups[year].append(fp)

        for year, files in sorted(annual_groups.items()):
            yield model_from_dir, year, sorted(files)


def find_representative_daily_file(appdata_dir: Path, model_from_dir: str, year: int) -> Path:
    """
    Find one daily FWI file for the given model and year.
    Metadata for yearly items is derived from this daily file, not Annual_Stat files.
    """
    model_dir = appdata_dir / model_from_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pattern = re.compile(
        rf"^{year}_(\d{{2}})_(\d{{2}})_T\d{{2}}_\d{{2}}_fwi\.nc$",
        re.IGNORECASE,
    )

    candidates: List[Path] = []
    for member_dir in sorted(model_dir.glob("*/member01")):
        if not member_dir.is_dir():
            continue
        for fp in sorted(member_dir.iterdir()):
            if fp.is_file() and fp.suffix.lower() == ".nc" and pattern.match(fp.name):
                candidates.append(fp)

    if not candidates:
        raise FileNotFoundError(
            f"No representative daily FWI file found for model={model_from_dir}, year={year}"
        )

    return candidates[0]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    with open("catalog_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eumet_id = cfg["id"]
    title = cfg.get("title", eumet_id)
    description = cfg.get("description", "")
    short_description = cfg.get("short_description")

    base_dir = Path.cwd()
    appdata_dir = Path(
        cfg.get("daily_data_dir", "/scratch/project_465000454/poladesu/ReRuns_FWI")
    )

    coll_dir = base_dir / eumet_id
    metadata_dir = coll_dir / "metadata"
    data_dir = coll_dir / "data"

    (metadata_dir / "items").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_bboxes: List[List[float]] = []

    # ---------------------------------------------------------------
    # 1. Stage monthly items from daily files
    # ---------------------------------------------------------------
    monthly_groups: Dict[Tuple[str, str, str, str, str, str, int, int], List[Path]] = defaultdict(list)

    for model_from_dir, fp in iter_daily_model_files(appdata_dir):
        m = DAILY_FWI_PATTERN.match(fp.name)
        if m is None:
            continue

        year = int(m.group("year"))
        month = int(m.group("month"))

        bbox, meta = read_nc_metadata(fp)

        meta["model"] = sanitize_item_token(meta.get("model", "").strip() or model_from_dir)
        meta["experiment"] = sanitize_item_token(normalize_experiment(meta.get("experiment", ""), year))
        meta["generation"] = sanitize_item_token(meta.get("generation", "").strip())
        meta["realization"] = sanitize_item_token(meta.get("realization", "").strip())
        meta["resolution"] = sanitize_item_token(meta.get("resolution", "").strip())
        meta["activity"] = sanitize_item_token(meta.get("activity", "").strip())
        meta["data_type"] = "monthly"

        require_metadata_fields(
            meta,
            ["model", "experiment", "generation", "realization", "resolution", "activity"],
            context=f"monthly file={fp}",
        )

        key = (
            meta["model"],
            meta["experiment"],
            meta["generation"],
            meta["realization"],
            meta["resolution"],
            meta["activity"],
            year,
            month,
        )
        monthly_groups[key].append(fp)
        all_bboxes.append(bbox)

    if not monthly_groups:
        raise RuntimeError(f"No daily FWI .nc files found under {appdata_dir}")

    for (model, experiment, generation, realization, resolution, activity, year, month), files in sorted(monthly_groups.items()):
        item_id = build_monthly_item_id(
            eumet_id=eumet_id,
            year=year,
            month=month,
            model=model,
            experiment=experiment,
            generation=generation,
            realization=realization,
            resolution=resolution,
            activity=activity,
            data_type="monthly",
        )

        item_folder = data_dir / str(year) / f"{month:02d}" / item_id
        item_folder.mkdir(parents=True, exist_ok=True)

        for fp in files:
            stage_file_with_hardlink(fp, item_folder / fp.name)

    # ---------------------------------------------------------------
    # 2. Stage yearly items from annual statistics per model
    #    Metadata is derived from representative DAILY files
    # ---------------------------------------------------------------
    for model_from_dir, year, files in iter_annual_model_files(appdata_dir):
        if not files:
            continue

        rep_daily_fp = find_representative_daily_file(appdata_dir, model_from_dir, year)
        bbox, meta = read_nc_metadata(rep_daily_fp)

        meta["model"] = sanitize_item_token(meta.get("model", "").strip() or model_from_dir)
        meta["experiment"] = sanitize_item_token(normalize_experiment(meta.get("experiment", ""), year))
        meta["generation"] = sanitize_item_token(meta.get("generation", "").strip())
        meta["realization"] = sanitize_item_token(meta.get("realization", "").strip())
        meta["resolution"] = sanitize_item_token(meta.get("resolution", "").strip())
        meta["activity"] = sanitize_item_token(meta.get("activity", "").strip())
        meta["data_type"] = "yearly"

        require_metadata_fields(
            meta,
            ["model", "experiment", "generation", "realization", "resolution", "activity"],
            context=f"yearly representative daily file={rep_daily_fp}",
        )

        all_bboxes.append(bbox)

        item_id = build_yearly_item_id(
            eumet_id=eumet_id,
            year=year,
            model=meta["model"],
            experiment=meta["experiment"],
            generation=meta["generation"],
            realization=meta["realization"],
            resolution=meta["resolution"],
            activity=meta["activity"],
            data_type=meta["data_type"],
        )

        # Keep MM folder level. Put yearly items in month "01".
        item_folder = data_dir / str(year) / "01" / item_id
        item_folder.mkdir(parents=True, exist_ok=True)

        for fp in files:
            stage_file_with_hardlink(fp, item_folder / fp.name)

    # ---------------------------------------------------------------
    # 3. Write collection metadata
    # ---------------------------------------------------------------
    if not all_bboxes:
        raise RuntimeError("No files were staged, so spatial extent could not be determined.")

    spatial_extent = [
        min(b[0] for b in all_bboxes),
        min(b[1] for b in all_bboxes),
        max(b[2] for b in all_bboxes),
        max(b[3] for b in all_bboxes),
    ]
    temporal_extent = (datetime(1990, 1, 1), datetime(2049, 12, 31))

    create_stac_collection(
        collection_path=metadata_dir / "collection.json",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        eumet_id=eumet_id,
        title=title,
        description=description,
        short_description=short_description,
    )
    create_collection_config(metadata_dir, eumet_id, spatial_extent)

    print(f"[OK] Staged collection in: {coll_dir}")
    print("Daily files staged as monthly items.")
    print("Annual statistics staged as yearly items under MM-level folder structure.")
    print("Yearly item metadata derived from representative daily files.")
    print("Item ids include model, experiment, generation, realization, resolution, activity, and data_type.")
    print("Hard-link staging enabled where possible; item_config.json is not created.")
    print("Next: python generate_item_metadata.py")


if __name__ == "__main__":
    main()
