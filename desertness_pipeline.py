# desertness_pipeline.py
import pathlib, yaml, numpy as np
import xarray as xr
from utils_terraclimate import fetch_terraclimate
from fuzzy_membership import s_shaped, z_shaped
from indices_climate import climate_mean, annual_sum, dryness_ratio_phi, aridity_index_ai, climatic_water_deficit_from_def
from semantics_superval import superval_masks_from_precisifications


def _load_cfg(path):
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))

def _maybe_write_geotiff(da: xr.DataArray, out_path: pathlib.Path):
    try:
        import rioxarray  # 可选依赖
        da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims("lon","lat")
        da.rio.to_raster(out_path.as_posix())
        print("GeoTIFF written:", out_path)
    except Exception as e:
        print("[warn] GeoTIFF export skipped:", e)

def _build_mask(outdir: pathlib.Path, P: xr.DataArray, PET: xr.DataArray):
    fy = outdir / "full_years_both.nc"
    if fy.exists():
        fy_da = xr.open_dataarray(fy)
        years_expected = int(fy_da.max().item())
        mask_full = (fy_da == years_expected)
        return mask_full

    return (P.notnull() & PET.notnull())

def run_desertness(cfg: dict, *, export_geotiff: bool = True):
    bbox   = tuple(cfg["region"]["bbox"])
    start, end = cfg["region"]["period"]
    outdir = pathlib.Path(cfg.get("output", {}).get("dir", "out"))
    outdir.mkdir(parents=True, exist_ok=True)

    ppt_y = annual_sum(fetch_terraclimate("ppt", (start, end), bbox, agg="none"))
    pet_y = annual_sum(fetch_terraclimate("pet", (start, end), bbox, agg="none"))
    P   = climate_mean(ppt_y).rename("P_mm_per_year")
    PET = climate_mean(pet_y).rename("PET_mm_per_year")

    mask = _build_mask(outdir, P, PET)
    P, PET = P.where(mask), PET.where(mask)

    eps = 1e-9
    phi = (PET / (P + eps)).where(mask).rename("phi_PET_over_P")
    ai  = (P / (PET + eps)).where(mask).rename("AI_P_over_PET")
    def_y = annual_sum(fetch_terraclimate("def", (start, end), bbox, agg="none"))
    CWD = climatic_water_deficit_from_def(def_y).where(mask)

    fz = cfg.get("fuzzy", {})
    ps = fz.get("phi_s", {"a": 2.0, "b": 3.0})
    az = fz.get("ai_z",  {"a": 0.30, "b": 0.60})
    cz = fz.get("cwd_s", {"a": 200.0, "b": 600.0})   
    mu_phi = xr.apply_ufunc(s_shaped, phi, ps["a"], ps["b"], dask="allowed")
    mu_ai  = xr.apply_ufunc(z_shaped,  ai,  az["a"], az["b"], dask="allowed")
    mu_cwd = xr.apply_ufunc(s_shaped, CWD, cz["a"], cz["b"], dask="allowed")
    combine_expr = fz.get("combine", None)
    if combine_expr:
        env = {"np": np, "mu_phi": mu_phi, "mu_ai": mu_ai, "mu_cwd": mu_cwd}
        out = eval(combine_expr, {"__builtins__": {}}, env)
        desertness = (out if isinstance(out, xr.DataArray)
                      else xr.DataArray(out, coords=mu_phi.coords, dims=mu_phi.dims))
    else:
        desertness = xr.ufuncs.maximum(mu_phi, 1.0 - mu_ai)
    desertness = desertness.where(mask).rename("desertness_mean")  
    print("[combine expr]", combine_expr or "DEFAULT: max(mu_phi, mu_ai)")
    print("[desertness min/max]", float(desertness.min()), float(desertness.max()))


    
    sv = cfg.get("supervaluation", None)
    if sv:
        valid_mask = xr.ufuncs.isfinite(phi) & xr.ufuncs.isfinite(ai) & xr.ufuncs.isfinite(CWD)
        supertrue_mask, superfalse_mask, undetermined_mask, votes_true, votes_total = \
            superval_masks_from_precisifications(phi, ai, CWD, sv, valid_mask=valid_mask)

        print("supervaluation applied (crisp across precisifications)")
        print("[K]", int(votes_total.max()), 
              "[ST%]", float(supertrue_mask.astype('int8').mean()*100),
              "[SF%]", float(superfalse_mask.astype('int8').mean()*100))
    else:
        supertrue_mask    = (desertness >= 0.7).rename("supertrue_mask")
        superfalse_mask   = (desertness <= 0.3).rename("superfalse_mask")
        undetermined_mask = (~supertrue_mask) & (~superfalse_mask)
        votes_true = votes_total = None

    P.to_netcdf(outdir/"P.nc"); PET.to_netcdf(outdir/"PET.nc")
    phi.to_netcdf(outdir/"phi.nc"); ai.to_netcdf(outdir/"ai.nc")
    desertness.to_netcdf(outdir/"desertness_mean.nc")
    undetermined_mask.to_netcdf(outdir/"undetermined_mask.nc")
    supertrue_mask.to_netcdf(outdir/"supertrue_mask.nc")
    
    if 'superfalse_mask' in locals() and (superfalse_mask is not None):
        superfalse_mask.to_netcdf(outdir/"superfalse_mask.nc")
    if votes_true is not None:
        votes_true.to_netcdf(outdir/"votes_true.nc")
        votes_total.to_netcdf(outdir/"votes_total.nc")

    if export_geotiff:
        _maybe_write_geotiff(desertness,outdir/"desertness_mean.tif")
        _maybe_write_geotiff(supertrue_mask.astype("uint8"), outdir/"supertrue_mask.tif")
        if 'superfalse_mask' in locals() and (superfalse_mask is not None):
            _maybe_write_geotiff(superfalse_mask.astype("uint8"), outdir/"superfalse_mask.tif")

    return {
        "P": P, "PET": PET, "phi": phi, "ai": ai,
        "desertness_mean": desertness,
        "supertrue_mask": supertrue_mask,
        "undetermined_mask": undetermined_mask,
    }


cfg = _load_cfg("config.yml")
out = run_desertness(cfg)
print("Done.")
