import xarray as xr
import rasterio
import numpy as np
from rasterio.enums import Resampling
from shapely.geometry import Polygon
import geopandas as gpd

class DesertPeakFinder:
    def __init__(self, desert_mask_path, terrain_tif_path):
        self.desert_mask = xr.open_dataarray(desert_mask_path)
        
        self.terrain_tif_path = terrain_tif_path
        self.terrain_data, self.transform, self.crs = self.read_terrain_data()

        self.aligned_desert_mask = self.align_desert_mask()

    def read_terrain_data(self):
        with rasterio.open(self.terrain_tif_path) as src:
            terrain_data = src.read(1, resampling=Resampling.nearest)
            transform = src.transform
            crs = src.crs
        return terrain_data, transform, crs

    def align_desert_mask(self):
        aligned_desert_mask = self.desert_mask.rio.write_crs(self.crs).rio.reproject(self.transform)
        return aligned_desert_mask

    def find_desert_peaks(self, elevation_threshold=600):
        desert_mask = self.aligned_desert_mask.where(self.aligned_desert_mask == 1, 0)  

       
        desert_peaks = (desert_mask == 1) & (self.terrain_data >= elevation_threshold)

        return desert_peaks

    def save_desert_peaks(self, output_tif_path):
        desert_peaks = self.find_desert_peaks()

        
        with rasterio.open(output_tif_path, 'w', driver='GTiff', height=desert_peaks.shape[0], width=desert_peaks.shape[1],
                           count=1, dtype=desert_peaks.dtype, crs=self.crs, transform=self.transform) as dst:
            dst.write(desert_peaks.astype(np.uint8), 1)


desert_mask_path = "out/desert_class_final.nc"  
terrain_tif_path = "out/15N000E_3S.tif"  


peak_finder = DesertPeakFinder(desert_mask_path, terrain_tif_path)

output_tif_path = "out/desert_peaks.tif"
peak_finder.save_desert_peaks(output_tif_path)

print(f"Desert peaks saved to {output_tif_path}")