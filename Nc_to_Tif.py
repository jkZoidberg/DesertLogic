import xarray as xr
import rioxarray

def convert_nc_to_tif(nc_file_path, tif_file_path):   
    ds = xr.open_dataset(nc_file_path)    
    da = ds["__xarray_dataarray_variable__"]   
    da.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    if not da.rio.crs:
        da.rio.write_crs("EPSG:4326", inplace=True)

    da.rio.to_raster(tif_file_path)
    print(f"GeoTIFF saved to {tif_file_path}")
nc_file_path = "out/desert_class_final.nc"  
tif_file_path = "out/odesert_class_final.tif"  
convert_nc_to_tif(nc_file_path, tif_file_path)
