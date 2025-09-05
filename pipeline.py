from __future__ import annotations

import pathlib
import yaml
import numpy as np
import xarray as xr
from typing import Dict, Optional, Tuple

from utils_terraclimate import fetch_terraclimate
from fuzzy_membership import s_shaped, z_shaped
from indices_climate import (
    annual_sum,
    climate_mean,
    climatic_water_deficit_from_def,
    dryness_ratio_phi,
    aridity_index_ai,
)
from semantics_superval import superval_masks_from_precisifications


# --------------------------- Config helpers ---------------------------

def load_config(path: str | pathlib.Path) -> dict:
    """Load YAML config as dict (no validation here; let callers validate if needed)."""
    p = pathlib.Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------- IO helpers ---------------------------

def _maybe_write_geotiff(da: xr.DataArray, out_path: pathlib.Path):
    """Try exporting a DataArray to GeoTIFF using rioxarray without failing the pipeline."""
    try:
        import rioxarray  # noqa: F401
        # try to infer dims (lat, lon) or (y, x)
        dims = list(da.dims)
        if ("lon" in dims and "lat" in dims):
            x_dim, y_dim = "lon", "lat"
        elif ("x" in dims and "y" in dims):
            x_dim, y_dim = "x", "y"
        else:
            # fallback to last two dims
            y_dim, x_dim = dims[-2], dims[-1]
        (da
         .rio.write_crs("EPSG:4326", inplace=True)
         .rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
         .rio.to_raster(out_path.as_posix()))
        print("GeoTIFF written:", out_path)
    except Exception as e:
        print("[warn] GeoTIFF export skipped:", e)


# --------------------------- Step 1: fetch & indices ---------------------------

def step1_fetch_and_indices(
    cfg: dict,
    *,
    write_nc: bool = True,
) -> Dict[str, xr.DataArray]:
    """
    Fetch TerraClimate and compute baseline climate indices.
    Returns P, PET, phi, ai, CWD and the validity mask used.
    """
    bbox: Tuple[float, float, float, float] = tuple(cfg["region"]["bbox"])  # (lat_min, lat_max, lon_min, lon_max)
    start, end = cfg["region"]["period"]
    outdir = pathlib.Path(cfg.get("output", {}).get("dir", "out"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Monthly → annual sum → multi-year mean
    ppt_y = annual_sum(fetch_terraclimate("ppt", (start, end), bbox, agg="none"))
    pet_y = annual_sum(fetch_terraclimate("pet", (start, end), bbox, agg="none"))
    P   = climate_mean(ppt_y).rename("P_mm_per_year")
    PET = climate_mean(pet_y).rename("PET_mm_per_year")

    # Validity mask (prefer full_years_both from coverage, else P & PET non-null)
    fy = outdir / "full_years_both.nc"
    if fy.exists():
        fy_da = xr.open_dataarray(fy)
        years_expected = int(fy_da.max().item())
        mask = (fy_da == years_expected)
    else:
        mask = P.notnull() & PET.notnull()

    P, PET = P.where(mask), PET.where(mask)

    # Indices (reuse helpers)
    phi = dryness_ratio_phi(PET, P).where(mask)  # φ = PET / P
    ai  = aridity_index_ai(P, PET).where(mask)   # AI = P / PET

    def_y = annual_sum(fetch_terraclimate("def", (start, end), bbox, agg="none"))
    CWD = climatic_water_deficit_from_def(def_y).where(mask)

    if write_nc:
        P.to_netcdf(outdir/"P.nc")
        PET.to_netcdf(outdir/"PET.nc")
        phi.to_netcdf(outdir/"phi.nc")
        ai.to_netcdf(outdir/"ai.nc")
        CWD.to_netcdf(outdir/"CWD.nc")

    return {"P": P, "PET": PET, "phi": phi, "ai": ai, "CWD": CWD, "mask": mask}


# --------------------------- Step 2: fuzzy combine ---------------------------

def step2_fuzzy(
    cfg: dict,
    *,
    phi: xr.DataArray,
    ai: xr.DataArray,
    CWD: xr.DataArray,
    write_nc: bool = True,
    export_geotiff: bool = False,
) -> xr.DataArray:
    """
    Build fuzzy memberships and combine according to cfg['fuzzy'].
    Returns the desertness membership in [0,1].
    """
    outdir = pathlib.Path(cfg.get("output", {}).get("dir", "out"))
    outdir.mkdir(parents=True, exist_ok=True)

    fz = cfg.get("fuzzy", {})
    ps = fz.get("phi_s", {"a": 2.0, "b": 3.0})
    az = fz.get("ai_z",  {"a": 0.30, "b": 0.60})
    cz = fz.get("cwd_s", {"a": 200.0, "b": 600.0})

    mu_phi = xr.apply_ufunc(s_shaped, phi, ps["a"], ps["b"], dask="allowed")
    mu_ai  = xr.apply_ufunc(z_shaped,  ai,  az["a"], az["b"], dask="allowed")
    mu_cwd = xr.apply_ufunc(s_shaped, CWD, cz["a"], cz["b"], dask="allowed")

    combine_expr = fz.get("combine")
    if combine_expr:
        env = {"np": np, "mu_phi": mu_phi, "mu_ai": mu_ai, "mu_cwd": mu_cwd}
        out = eval(combine_expr, {"__builtins__": {}}, env)
        D = (out if isinstance(out, xr.DataArray) else xr.DataArray(out, coords=mu_phi.coords, dims=mu_phi.dims))
    else:
        D = (0.5 * mu_phi) + (0.3 * mu_cwd) + (0.2 * mu_ai)

    D = D.rename("desertness_mean")

    if write_nc:
        D.to_netcdf(outdir/"desertness_mean.nc")
    if export_geotiff:
        _maybe_write_geotiff(D, outdir/"desertness_mean.tif")

    return D


# --------------------------- Step 3: supervaluation ---------------------------

def step3_supervaluation(
    cfg: dict,
    *,
    phi: xr.DataArray,
    ai: xr.DataArray,
    CWD: xr.DataArray,
    desertness: xr.DataArray,
    write_nc: bool = True,
    export_geotiff: bool = False,
) -> Dict[str, Optional[xr.DataArray]]:
    """
    Apply supervaluation if cfg['supervaluation'] is present. Otherwise, threshold fuzzy desertness.
    Returns dict with ST, SF, UD and optional votes_true/total.
    """
    outdir = pathlib.Path(cfg.get("output", {}).get("dir", "out"))
    outdir.mkdir(parents=True, exist_ok=True)

    sv_cfg = cfg.get("supervaluation")
    votes_true = votes_total = None

    if sv_cfg:
        valid_mask = xr.ufuncs.isfinite(phi) & xr.ufuncs.isfinite(ai) & xr.ufuncs.isfinite(CWD)
        ST, SF, UD, votes_true, votes_total = superval_masks_from_precisifications(
            phi, ai, CWD, sv_cfg, valid_mask=valid_mask
        )
    else:
        ST = (desertness >= 0.7).rename("supertrue_mask")
        SF = (desertness <= 0.3).rename("superfalse_mask")
        UD = (~ST) & (~SF)

    if write_nc:
        ST.to_netcdf(outdir/"supertrue_mask.nc")
        UD.to_netcdf(outdir/"undetermined_mask.nc")
        try:
            SF.to_netcdf(outdir/"superfalse_mask.nc")
        except Exception:
            pass
        if votes_true is not None:
            votes_true.to_netcdf(outdir/"votes_true.nc")
            votes_total.to_netcdf(outdir/"votes_total.nc")
    if export_geotiff:
        _maybe_write_geotiff(ST.astype("uint8"), outdir/"supertrue_mask.tif")
        try:
            _maybe_write_geotiff(SF.astype("uint8"), outdir/"superfalse_mask.tif")
        except Exception:
            pass

    return {
        "supertrue_mask": ST,
        "superfalse_mask": SF,
        "undetermined_mask": UD,
        "votes_true": votes_true,
        "votes_total": votes_total,
    }


# --------------------------- Orchestrator (compat) ---------------------------

def run_desertness(cfg: dict, *, export_geotiff: bool = True) -> Dict[str, xr.DataArray]:
    """Backward compatible end-to-end run, built on the stepwise API."""
    step1 = step1_fetch_and_indices(cfg, write_nc=True)
    D = step2_fuzzy(cfg, phi=step1["phi"], ai=step1["ai"], CWD=step1["CWD"], write_nc=True, export_geotiff=export_geotiff)
    sv = step3_supervaluation(cfg, phi=step1["phi"], ai=step1["ai"], CWD=step1["CWD"], desertness=D,
                              write_nc=True, export_geotiff=export_geotiff)
    return {
        "P": step1["P"],
        "PET": step1["PET"],
        "phi": step1["phi"],
        "ai": step1["ai"],
        "desertness_mean": D,
        "supertrue_mask": sv["supertrue_mask"],
        "undetermined_mask": sv["undetermined_mask"],
    }


# --------------------------- CLI (optional) ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run DesertLogic stepwise pipeline")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--no-geotiff", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = run_desertness(cfg, export_geotiff=not args.no_geotiff)
    print("Done.")
