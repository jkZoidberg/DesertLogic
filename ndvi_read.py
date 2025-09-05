import ee, geemap
from pathlib import Path

ee.Initialize(project="chicago-crime-441523")
roi = ee.Geometry.Rectangle([-10.0, 0.0, 10.0, 28.0])

ndvi_raw = (ee.ImageCollection("MODIS/061/MOD13A3")
            .select("NDVI")
            .filterBounds(roi)
            .filterDate("2025-07-01", "2025-07-31")
            .mean())

# transfer to int16：NDVI(0–1) -> 0–10000
ndvi_int16 = (ndvi_raw.multiply(0.0001)           # 0–1
              .multiply(10000)                    # 0–10000
              .clamp(0, 10000)
              .toInt16()
              .rename("NDVI"))

out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
out_tif = out_dir / "NDVI_202507_WGS84_int16.tif"

geemap.ee_export_image(
    ee_object=ndvi_int16,
    filename=str(out_tif),
    region=roi,
    crs="EPSG:4326",
    scale=1000,
    timeout=600 
)

print(f"[OK] 本地已保存: {out_tif.resolve()}")
#把 0–1 的浮点 NDVI 放大到 0–10000，并转为 int16。体积立刻减半以上，通常就能过 48 MiB 限制。
#读取时记得除以 10000 恢复到 0–1：ndvi = rxr.open_rasterio(...)/10000.0