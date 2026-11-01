"""
Lake Geometry and Spatial ROI Module
====================================
Provides verified spatial polygon geometries and search envelopes for
monitored Himalayan glacial lakes, derived from the authoritative GLO v1.02
inventory (2023 baseline).

Supports:
- Projected coordinate transformation (e.g. to raster CRS).
- Metric buffering (0m core prior vs 50m search envelope in UTM EPSG:32645).
- Rasterization of spatial ROIs onto satellite imagery grids.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
from pyproj import Transformer
import rasterio.features
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


DATA_DIR = Path(__file__).resolve().parent / "data"
GEOJSON_PATH = DATA_DIR / "lake_polygons.geojson"

# Canonical UTM coordinate reference system for Nepal Himalayas
UTM_CRS = "EPSG:32645"

# Cache for loaded geometries in WGS84
_CACHED_LAKE_GEODATAFRAME: Optional[gpd.GeoDataFrame] = None


def get_lake_geodataframe() -> gpd.GeoDataFrame:
    """Load and return the verified lake polygon GeoDataFrame (EPSG:4326)."""
    global _CACHED_LAKE_GEODATAFRAME
    if _CACHED_LAKE_GEODATAFRAME is None:
        if not GEOJSON_PATH.is_file():
            raise FileNotFoundError(
                f"Lake polygons GeoJSON not found at: {GEOJSON_PATH}"
            )
        gdf = gpd.read_file(GEOJSON_PATH)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        _CACHED_LAKE_GEODATAFRAME = gdf
    return _CACHED_LAKE_GEODATAFRAME


def get_lake_geometry(
    lake_id: str,
    buffer_m: float = 0.0,
    target_crs: Optional[str] = None,
) -> BaseGeometry:
    """Retrieve verified lake geometry, optionally buffered and transformed.

    Parameters
    ----------
    lake_id : str
        Canonical lake identifier (e.g. 'imja_tsho', 'thulagi').
    buffer_m : float, default=0.0
        Buffer distance in meters applied in metric UTM projection.
    target_crs : Optional[str]
        Target coordinate reference system (e.g. raster CRS).
        If None, returns geometry in EPSG:4326.

    Returns
    -------
    BaseGeometry
        Shapely polygon or multipolygon.
    """
    gdf = get_lake_geodataframe()
    lid = str(lake_id).lower().strip()
    match = gdf[gdf["lake_id"] == lid]
    if match.empty:
        raise KeyError(
            f"Lake '{lake_id}' not found in verified lake polygons. "
            f"Available: {sorted(gdf['lake_id'].tolist())}"
        )

    geom_4326 = match.geometry.iloc[0]

    # Transform to metric UTM to apply accurate distance buffer in meters
    transformer_to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    geom_utm = shapely_transform(transformer_to_utm.transform, geom_4326)

    if buffer_m != 0.0:
        geom_utm = geom_utm.buffer(buffer_m)

    if target_crs is None or target_crs.upper() == "EPSG:4326":
        transformer_back = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
        return shapely_transform(transformer_back.transform, geom_utm)

    if target_crs.upper() == UTM_CRS:
        return geom_utm

    transformer_to_target = Transformer.from_crs(UTM_CRS, target_crs, always_xy=True)
    return shapely_transform(transformer_to_target.transform, geom_utm)


def rasterize_lake_roi(
    lake_id: str,
    out_shape: Tuple[int, int],
    transform: Any,
    crs: str,
    buffer_m: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rasterize 0m core prior and buffered search envelope onto an image grid.

    Parameters
    ----------
    lake_id : str
        Lake identifier.
    out_shape : Tuple[int, int]
        (height, width) of the target raster.
    transform : Affine
        Affine geotransform of the target raster.
    crs : str
        CRS string of the target raster.
    buffer_m : float, default=50.0
        Expansion envelope buffer in meters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (m0, m_buf, pixel_area_km2):
        - m0: Boolean mask of core 0m lake polygon.
        - m_buf: Boolean mask of buffered search envelope.
        - pixel_area_km2: Area of a single pixel in km².
    """
    poly_0 = get_lake_geometry(lake_id, buffer_m=0.0, target_crs=crs)
    poly_buf = get_lake_geometry(lake_id, buffer_m=buffer_m, target_crs=crs)

    m0 = rasterio.features.rasterize(
        [(poly_0, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ) == 1

    m_buf = rasterio.features.rasterize(
        [(poly_buf, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ) == 1

    pixel_area_km2 = abs(transform.a * transform.e) / 1e6

    return m0, m_buf, pixel_area_km2
