from OasisFinder import OasisFinder
from pathlib import Path

ndvi_tif = "out/NDVI_202507_WGS84_int16.tif"  
desert_tif = "out/odesert_class_final.tif"              

finder = OasisFinder(ndvi_tif, desert_tif)
finder.load()
finder.compute_oasis(ndvi_threshold=0.20, min_pixels=50, connectivity=8)

Path("out").mkdir(exist_ok=True)
finder.save_raster("out/oasis_mask_202507.tif")
finder.save_vectors("out/oasis_polygons_202507.geojson", dissolve=False, centroids=False)

print(finder.quick_stats())
print("Done.")