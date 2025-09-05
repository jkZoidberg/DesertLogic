import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict


_BASE = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
_VAR_URL: Dict[str, str] = {
    v: f"{_BASE}agg_terraclimate_{v}_1958_CurrentYear_GLOBE.nc"
    for v in [
        "ppt", "pet", "aet", "def", "soil", "runoff", "srad",
        "tmax", "tmin", "tmean", "vap", "vpd", "ws",
        "pdsi", "swe",
        "ro",
    ]
}

def _open_remote(url: str) -> xr.Dataset:
    try:
        return xr.open_dataset(url, engine="netcdf4")
    except Exception:
        url_pydap = url.replace("http://", "dap2://")
        return xr.open_dataset(url_pydap, engine="pydap")

def _lat_slice(lat_da: xr.DataArray, lat_min: float, lat_max: float):
    if float(lat_da[0]) > float(lat_da[-1]):  
        return slice(lat_max, lat_min)
    return slice(lat_min, lat_max)

def _apply_agg(
    da: xr.DataArray,
    agg: Optional[Literal["none", "annual_mean", "annual_sum", "climatology_monthly"]] = "none",
) -> xr.DataArray:
    if agg in (None, "none") or "time" not in da.dims:
        return da
    if agg == "annual_mean":
        return da.groupby("time.year").mean("time", keep_attrs=True)
    if agg == "annual_sum":
        return da.groupby("time.year").sum("time", keep_attrs=True)
    if agg == "climatology_monthly":
        return da.groupby("time.month").mean("time", keep_attrs=True)
    raise ValueError(f"Unknown agg='{agg}'")

def fetch_terraclimate(
    var: str,
    time_range: Tuple[str, str],
    bbox: Tuple[float, float, float, float],
    *,
    agg: Optional[Literal["none", "annual_mean", "annual_sum", "climatology_monthly"]] = "none",
    to_netcdf: Optional[str] = None,
    to_geotiff: Optional[str] = None,
    geotiff_one_file_per_step: bool = True,
    auto_load: bool = False,
) -> xr.DataArray:
    if var not in _VAR_URL:
        raise ValueError(f"Unknown var '{var}'. Available: {sorted(_VAR_URL)}")
    url = _VAR_URL[var]

    ds = _open_remote(url)

    for c in ("time", "lat", "lon"):
        if c not in ds.coords:
            raise KeyError(f"Missing coord '{c}'. Have: {list(ds.coords)}")

    t0, t1 = time_range
    ds = ds.sel(time=slice(t0, t1))

    lat_min, lat_max, lon_min, lon_max = bbox
    lat_sel = _lat_slice(ds["lat"], lat_min, lat_max)

    if lon_min <= lon_max:
        sub = ds.sel(lat=lat_sel, lon=slice(lon_min, lon_max))
    else:
        sub1 = ds.sel(lat=lat_sel, lon=slice(lon_min, 180.0))
        sub2 = ds.sel(lat=lat_sel, lon=slice(-180.0, lon_max))
        sub = xr.combine_by_coords([sub1, sub2])

    if var in sub.data_vars:
        da = sub[var]
    elif var == "runoff" and "ro" in sub.data_vars:
        da = sub["ro"]
    elif var == "ro" and "runoff" in sub.data_vars:
        da = sub["runoff"]
    else:
        dvars = list(sub.data_vars)
        if len(dvars) == 1:
            da = sub[dvars[0]]
        else:
            raise KeyError(f"Variable '{var}' not found in subset. Available: {dvars}")

    da = _apply_agg(da, agg=agg)

    try:
        import rioxarray  # noqa: F401
        da = da.rio.write_crs("EPSG:4326", inplace=False)
        if "lon" in da.dims and "lat" in da.dims:
            da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    except Exception:
        pass

    if to_netcdf:
        Path(to_netcdf).parent.mkdir(parents=True, exist_ok=True)
        da.to_netcdf(to_netcdf)

    if to_geotiff:
        try:
            import rioxarray  # noqa: F401
        except Exception as e:
            raise ImportError("need rioxarray：pip install rioxarray") from e

        out = Path(to_geotiff)
        out.parent.mkdir(parents=True, exist_ok=True)

        if "time" in da.dims and da.sizes.get("time", 1) > 1:
            if not geotiff_one_file_per_step:
                raise ValueError("set geotiff_one_file_per_step=True ")
            for t in da["time"].values:
                ts = np.datetime_as_string(t, unit="M" if agg == "climatology_monthly" else "D")
                da.sel(time=t).rio.to_raster(out.with_name(out.stem + f"_{ts}.tif").as_posix())
        else:
            da.rio.to_raster(out.as_posix())

    if auto_load:
        da = da.load()

    return da

def open_terraclimate_dataset(var: str) -> xr.Dataset:
    if var not in _VAR_URL:
        raise ValueError(f"Unknown var '{var}'. Available: {sorted(_VAR_URL)}")
    return _open_remote(_VAR_URL[var])

