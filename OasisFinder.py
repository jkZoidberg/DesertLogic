# oasis_finder.py
from pathlib import Path
import json
import numpy as np
import rioxarray as rxr
import xarray as xr
import rasterio
from rasterio.features import sieve, shapes
from shapely.geometry import shape, mapping
import geopandas as gpd

class OasisFinder:

    def __init__(self, ndvi_tif: str, desert_mask_tif: str):
        self.ndvi_path = Path(ndvi_tif)
        self.desert_path = Path(desert_mask_tif)
        self.ndvi = None            # xarray.DataArray
        self.desert = None          # xarray.DataArray
        self.oasis = None           # xarray.DataArray (uint8, 1=oasis, 0=non-oasis)

    def load(self):
        # Read rasters (band-1) with rioxarray
        self.ndvi = rxr.open_rasterio(self.ndvi_path).squeeze("band", drop=True)
        self.desert = rxr.open_rasterio(self.desert_path).squeeze("band", drop=True)

        # Ensure CRS present
        if self.ndvi.rio.crs is None or self.desert.rio.crs is None:
            raise ValueError("NDVI or desert mask has no CRS. Please ensure GeoTIFFs are georeferenced.")

        # Align NDVI grid to desert grid (reproject & resample)
        if not self._grids_match():
            self.ndvi = self.ndvi.rio.reproject_match(self.desert)

        # Normalize NDVI if it looks like int16 0–10000
        self.ndvi = self._normalize_ndvi(self.ndvi)

        # Force desert to 0/1 and mask invalids
        self.desert = xr.where(self.desert.fillna(0) > 0.5, 1, 0).astype("uint8")

    def _grids_match(self) -> bool:
        # Same CRS and identical transform/shape?
        try:
            return (
                (self.ndvi.rio.crs == self.desert.rio.crs) and
                (self.ndvi.rio.transform() == self.desert.rio.transform()) and
                (self.ndvi.shape == self.desert.shape)
            )
        except Exception:
            return False

    def _normalize_ndvi(self, da: xr.DataArray) -> xr.DataArray:
        # If values exceed 1.5 or dtype is int16, assume scale factor 1/10000
        vmin = float(da.quantile(0.01).values)
        vmax = float(da.quantile(0.99).values)
        if da.dtype == np.int16 or vmax > 1.5:
            da = da.astype("float32") / 10000.0
        # Clip to [0,1] and keep nodata as is
        da = xr.where(np.isfinite(da), da.clip(0, 1), da)
        return da

    def compute_oasis(self, ndvi_threshold: float = 0.20, min_pixels: int = 50, connectivity: int = 8):
        if self.ndvi is None or self.desert is None:
            raise RuntimeError("Call load() first.")

        oasis_bool = (self.desert == 1) & (self.ndvi >= ndvi_threshold)
        oasis_u8 = xr.where(oasis_bool, 1, 0).astype("uint8")

        # Remove tiny patches using rasterio.features.sieve (needs numpy array)
        arr = oasis_u8.values
        if arr.ndim != 2:
            raise ValueError("Unexpected raster shape. Expect 2D array.")

        transform = self.desert.rio.transform()
        # sieve expects uint8 labels; returns cleaned array
        cleaned = sieve(arr, size=min_pixels, connectivity=connectivity)

        self.oasis = xr.DataArray(
            cleaned.astype("uint8"),
            coords=self.desert.coords,
            dims=self.desert.dims,
            name="oasis"
        ).rio.write_crs(self.desert.rio.crs).rio.write_transform(transform)

    def save_raster(self, out_tif: str):
        if self.oasis is None:
            raise RuntimeError("Call compute_oasis() first.")
        out_p = Path(out_tif)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Preserve georeferencing via rioxarray
        self.oasis.rio.to_raster(out_p.as_posix(), compress="DEFLATE", dtype="uint8")

    def save_vectors(self, out_geojson: str, dissolve: bool = False, centroids: bool = False):
        if self.oasis is None:
            raise RuntimeError("Call compute_oasis() first.")

        out_p = Path(out_geojson)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        arr = self.oasis.values
        transform = self.oasis.rio.transform()
        crs = self.oasis.rio.crs

        # Extract vector features where oasis==1
        feats = []
        for geom, val in shapes(arr, mask=(arr == 1), transform=transform, connectivity=8):
            if val == 1:
                geom_obj = shape(geom)
                if geom_obj.is_empty:
                    continue
                feats.append(geom_obj)

        if not feats:
            # Empty collection
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
            return

        gdf = gpd.GeoDataFrame({"class": 1}, geometry=feats, crs=crs)

        if centroids:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.centroid

        if dissolve and not centroids:
            gdf = gdf.dissolve(by="class", as_index=False)

        gdf.to_file(out_p.as_posix(), driver="GeoJSON")

    def quick_stats(self) -> dict:
        """
        Return basic area stats in pixels.
        """
        if self.oasis is None:
            raise RuntimeError("Call compute_oasis() first.")
        total = int(np.isfinite(self.desert.values).sum())
        oasis_px = int((self.oasis.values == 1).sum())
        desert_px = int((self.desert.values == 1).sum())
        return {
            "pixels_total_valid": total,
            "pixels_desert": desert_px,
            "pixels_oasis": oasis_px,
            "oasis_in_desert_ratio": float(oasis_px) / max(desert_px, 1)
        }