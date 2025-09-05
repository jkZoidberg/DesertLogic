import pathlib
import xarray as xr
import rioxarray
from rdr import rule_ud_to_stweak, rule_ud_to_sfweak, rule_ud_to_st_neighborhood, rule_ud_to_sf_neighborhood

def _maybe_write_geotiff(da: xr.DataArray, out_path: pathlib.Path):
    try:
        import rioxarray  
        lat_name, lon_name = da.dims[-2], da.dims[-1]
        (da.astype("uint8")
           .rio.write_crs("EPSG:4326", inplace=True)
           .rio.set_spatial_dims(lon_name, lat_name, inplace=True)
           .rio.to_raster(out_path.as_posix()))
        print("GeoTIFF written:", out_path)
    except Exception as e:
        print("[warn] GeoTIFF export skipped:", e)

def run_phase1(outdir="out",
               st_thr: float = 0.75,
               sf_thr: float = 0.25,
               export_geotiff: bool = False):
    outdir = pathlib.Path(outdir)
    ud_path  = outdir / "undetermined_mask.nc"
    vt_path  = outdir / "votes_true.nc"
    vtt_path = outdir / "votes_total.nc"
    missing = [p.name for p in [ud_path, vt_path, vtt_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺少必要输入：{missing}；请在 supervaluation 阶段保存 votes_true/votes_total。")

    UD  = xr.open_dataarray(ud_path).astype(bool)
    VT  = xr.open_dataarray(vt_path)
    VTT = xr.open_dataarray(vtt_path)

    # Phase 1 
    stweak = rule_ud_to_stweak(UD, VT, VTT, thr=st_thr)
    sfweak = rule_ud_to_sfweak(UD, VT, VTT, thr=sf_thr)

    #  NetCDF
    stweak.to_netcdf(outdir / "rdr_ST_weak_mask.nc")
    sfweak.to_netcdf(outdir / "rdr_SF_weak_mask.nc")

    def _pct(A: xr.DataArray) -> float:
        den = int((UD | stweak | sfweak).astype("int8").sum())
        return float((A.astype("int8").sum() / max(den, 1)) * 100.0)

    print(f"[RDR Phase1] ST_WEAK%={_pct(stweak):.2f}  SF_WEAK%={_pct(sfweak):.2f}")

    if export_geotiff:
        _maybe_write_geotiff(stweak, outdir / "rdr_ST_weak_mask.tif")
        _maybe_write_geotiff(sfweak, outdir / "rdr_SF_weak_mask.tif")

def run_phase2(outdir="out", tau_strict=0.70, tau_mid=0.40, export_geotiff=True):
    import pathlib
    outdir = pathlib.Path(outdir)
    UD     = xr.open_dataarray(outdir/"undetermined_mask.nc").astype(bool)
    p_arid = xr.open_dataarray(outdir/"p_arid.nc")

    st_str = rule_ud_to_st_neighborhood(UD, p_arid, tau_strict=tau_strict)
    sf_str = rule_ud_to_sf_neighborhood(UD, p_arid, tau_mid=tau_mid)

    st_str.to_netcdf(outdir/"rdr_ST_strong_neigh.nc")
    sf_str.to_netcdf(outdir/"rdr_SF_strong_neigh.nc")

    if export_geotiff:
        try:
            import rioxarray
            for da, name in [(st_str, "rdr_ST_strong_neigh.tif"),
                             (sf_str, "rdr_SF_strong_neigh.tif")]:
                lat_name, lon_name = da.dims[-2], da.dims[-1]
                (da.astype("uint8")
                    .rio.write_crs("EPSG:4326", inplace=True)
                    .rio.set_spatial_dims(lon_name, lat_name, inplace=True)
                    .rio.to_raster((outdir/name).as_posix()))
        except Exception as e:
            print("[warn] GeoTIFF export skipped:", e)

def finalize_labels(outdir="out"):
    import numpy as np, pathlib
    outdir = pathlib.Path(outdir)
    ST = xr.open_dataarray(outdir/"supertrue_mask.nc").astype(bool)
    SF = xr.open_dataarray(outdir/"superfalse_mask.nc").astype(bool)
    UD = xr.open_dataarray(outdir/"undetermined_mask.nc").astype(bool)

    # RDR Phase1
    st_w = xr.open_dataarray(outdir/"rdr_ST_weak_mask.nc").astype(bool)
    sf_w = xr.open_dataarray(outdir/"rdr_SF_weak_mask.nc").astype(bool)
    try:
        st_s = xr.open_dataarray(outdir/"rdr_ST_strong_neigh.nc").astype(bool)
        sf_s = xr.open_dataarray(outdir/"rdr_SF_strong_neigh.nc").astype(bool)
    except Exception:
        st_s = xr.zeros_like(ST, dtype=bool)
        sf_s = xr.zeros_like(SF, dtype=bool)


    final_ST = ST | st_s | (st_w & ~sf_s & ~sf_w)
    final_SF = SF | sf_s | (sf_w & ~st_s & ~st_w)
    final_UD = ~(final_ST | final_SF) & (ST | SF | UD)


    cls = xr.zeros_like(ST, dtype="uint8").rename("desert_class_final")
    cls = xr.where(final_SF, 0, cls)
    cls = xr.where(final_ST, 1, cls)
    cls = xr.where(final_UD, 2, cls)
    cls.to_netcdf(outdir/"desert_class_final.nc")
    print("Final %:", {
        "ST": int(final_ST.astype("int8").sum().item()),
        "SF": int(final_SF.astype("int8").sum().item()),
        "UD": int(final_UD.astype("int8").sum().item()),
    })


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



run_phase1(outdir="out", st_thr=0.75, sf_thr=0.25, export_geotiff=True)
run_phase2(outdir="out", tau_strict=0.70, tau_mid=0.40, export_geotiff=True)
finalize_labels(outdir="out")