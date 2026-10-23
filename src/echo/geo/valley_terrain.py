"""Real metric DEM statistics at supplied georeferenced river/valley patch centres."""

import numpy as np

TERRAIN_FEATURE_NAMES = ["elevation_p05_km", "elevation_p50_km", "elevation_p95_km",
                         "relative_river_height_p05_km", "relative_river_height_p50_km",
                         "relative_river_height_p95_km", "slope_p50_degrees", "slope_p95_degrees",
                         "low_terrain_fraction_below_10m_relative", "relative_x_km", "relative_y_km",
                         "along_river_km", "neighborhood_radius_km", "valid_DEM_coverage"]


def patch_statistics(elevation, circle_mask, pixel_m, river_reference_m, x_km, y_km,
                     along_river_km, radius_km, minimum_coverage=.9):
    if pixel_m <= 0 or not np.isfinite(river_reference_m) or not circle_mask.any():
        raise ValueError("DEM分辨率、河网参考高程或邻域无效")
    valid = np.isfinite(elevation) & circle_mask
    coverage = valid.sum() / circle_mask.sum()
    if coverage < minimum_coverage:
        return np.zeros(len(TERRAIN_FEATURE_NAMES), dtype="float32"), True
    dy, dx = np.gradient(elevation.astype("float64"), pixel_m)
    slope = np.rad2deg(np.arctan(np.hypot(dx, dy)))
    valid_slope = valid & np.isfinite(slope)
    if not valid_slope.any():
        return np.zeros(len(TERRAIN_FEATURE_NAMES), dtype="float32"), True
    z = elevation[valid]
    quantiles = np.percentile(z, [5, 50, 95])
    slope_quantiles = np.percentile(slope[valid_slope], [50, 95])
    token = [*(quantiles / 1000), *((quantiles - river_reference_m) / 1000),
             *slope_quantiles, float(np.mean(z - river_reference_m <= 10)),
             x_km, y_km, along_river_km, radius_km, coverage]
    return np.asarray(token, dtype="float32"), False


def extract_valley_tokens(dem_path, centres, metric_crs, station_xy, vertical_unit,
                          vertical_datum, resolution_m=30):
    """centres carry verified metric river position, along-river offset and DEM reference.

    Centre locations must come from a mapped river/reach geometry, not polygon
    centroids. Token radii are selected explicitly from 1,5,10 km.
    """
    import rasterio
    from pyproj import CRS
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import calculate_default_transform
    from rasterio.windows import Window, from_bounds

    crs = CRS.from_user_input(metric_crs)
    if not crs.is_projected or any(axis.unit_name != "metre" for axis in crs.axis_info):
        raise ValueError("局地DEM运算必须在已声明的米制投影下进行")
    if len(centres) != 16 or vertical_unit != "m" or not vertical_datum:
        raise ValueError("需要16个已定位河谷片段，并明确DEM垂向单位与基准")
    tokens, masks = [], []
    with rasterio.open(dem_path) as source:
        transform, width, height = calculate_default_transform(
            source.crs, crs, source.width, source.height, *source.bounds, resolution=resolution_m)
        with WarpedVRT(source, crs=crs, transform=transform, width=width, height=height, nodata=np.nan,
                       dtype="float32") as dem:
            for centre in centres:
                x, y, radius = centre["x_m"], centre["y_m"], centre["radius_km"] * 1000
                if radius not in {1000, 5000, 10000} or centre.get("position_source") != "mapped_river":
                    raise ValueError("局地片段必须使用真实河网位置及1/5/10km邻域")
                window = from_bounds(x - radius, y - radius, x + radius, y + radius, dem.transform)
                window = window.round_offsets().round_lengths()
                arr = np.full((int(window.height), int(window.width)), np.nan, dtype="float32")
                try:
                    clipped = window.intersection(Window(0, 0, dem.width, dem.height))
                    values = dem.read(1, window=clipped, masked=True).filled(np.nan)
                    top, left = int(clipped.row_off - window.row_off), int(clipped.col_off - window.col_off)
                    arr[top:top + values.shape[0], left:left + values.shape[1]] = values
                except rasterio.errors.WindowError:
                    pass
                transform = dem.window_transform(window)
                rr, cc = np.indices(arr.shape)
                xx, yy = rasterio.transform.xy(transform, rr, cc)
                circle = (np.asarray(xx).reshape(arr.shape) - x)**2 + (np.asarray(yy).reshape(arr.shape) - y)**2 <= radius**2
                token, mask = patch_statistics(
                    arr, circle, abs(transform.a), centre["river_reference_m"],
                    (x - station_xy[0]) / 1000, (y - station_xy[1]) / 1000,
                    centre["along_river_m"] / 1000, radius / 1000)
                tokens.append(token)
                masks.append(mask)
    return np.asarray(tokens), np.asarray(masks)
