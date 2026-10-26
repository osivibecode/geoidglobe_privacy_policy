"""Builds the bundled geoid and time-zone grids used by the app.

Inputs (download once):
  egm08_25.tif                    EGM2008 geoid undulation, 2.5' grid
                                  https://cdn.proj.org/us_nga_egm08_25.tif
  timezones-with-oceans.geojson   https://github.com/evansiroky/timezone-boundary-builder/releases

Outputs in the given directory (shared/src/commonMain/composeResources/files):
  geoid_cm.bin       int16 little-endian, geoid height in centimetres, 361 rows x 720 cols
  zone_index.bin     uint16 little-endian, index into zone_names.txt, same grid
  zone_names.txt     one IANA time zone id per line

Grid: row 0 is latitude +90, row 360 is -90; column 0 is longitude -180. Step 0.5 degrees.

Usage: python3 tools/build_data.py <egm08_25.tif> <timezones.geojson> <output dir>
Needs: numpy, shapely, tifffile, imagecodecs
"""

import json
import os
import sys

import numpy as np
import shapely
import tifffile

STEP = 0.5
ROWS = int(round(180 / STEP)) + 1  # latitude 90 -> -90
COLS = int(round(360 / STEP))  # longitude -180 -> 179.5
PIXELS_PER_DEGREE = 24  # 2.5 arc-minute grid
STRIDE = int(round(STEP * PIXELS_PER_DEGREE))  # 12


def geoid_grid(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        raw = tif.pages[0].asarray()
    sampled = raw[::STRIDE, ::STRIDE][:ROWS, :COLS].astype(np.float64)
    assert sampled.shape == (ROWS, COLS), sampled.shape
    return sampled


def time_zone_grid(geojson_path):
    lats = 90 - STEP * np.arange(ROWS)
    lons = -180 + STEP * np.arange(COLS)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    px = lon_grid.ravel()
    py = lat_grid.ravel()

    with open(geojson_path, encoding="utf-8") as f:
        features = json.load(f)["features"]

    names = []
    zone = np.full(px.shape, -1, dtype=np.int32)
    for feature in features:
        name = feature["properties"]["tzid"]
        index = len(names)
        names.append(name)
        geometry = shapely.geometry.shape(feature["geometry"])
        minx, miny, maxx, maxy = geometry.bounds
        candidates = np.where((px >= minx) & (px <= maxx) & (py >= miny) & (py <= maxy))[0]
        if candidates.size == 0:
            continue
        shapely.prepare(geometry)
        # intersects (not contains) so nodes on a shared border still get a zone
        inside = shapely.intersects_xy(geometry, px[candidates], py[candidates])
        zone[candidates[inside]] = index

    uncovered = int((zone < 0).sum())
    if uncovered:
        raise SystemExit(f"{uncovered} grid nodes have no time zone")
    print(f"time zones: {len(names)} ids")
    return names, zone.reshape(ROWS, COLS)


def main(tif_path, geojson_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    geoid = geoid_grid(tif_path)
    centimetres = np.round(geoid * 100).astype("<i2")
    names, zones = time_zone_grid(geojson_path)
    print(f"geoid: min {geoid.min():.2f} m, max {geoid.max():.2f} m")

    centimetres.tofile(os.path.join(out_dir, "geoid_cm.bin"))
    zones.astype("<u2").tofile(os.path.join(out_dir, "zone_index.bin"))
    with open(os.path.join(out_dir, "zone_names.txt"), "w", encoding="utf-8") as out:
        out.write("\n".join(names) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
