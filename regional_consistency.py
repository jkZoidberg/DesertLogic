
import numpy as np
import xarray as xr
from typing import Dict, Optional
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

def compute_regional_labels(
    D: xr.DataArray,                 # desertness_mean, 0-1
    ST: xr.DataArray,                # supertrue_mask (bool/0-1)
    UD: xr.DataArray,                # undetermined_mask (bool/0-1)
    *,
    k: int = 11,                     
    thr_pixel: float = 0.70,         
    tau_strict: float = 0.70,       
    tau_mid: float = 0.40,          
    elev: Optional[xr.DataArray] = None,    
    montane_med_elev: float = 1500.0,
    montane_relief: float = 500.0,
) -> Dict[str, xr.DataArray]:
    
    
    STb = ST.astype(bool)
    UDb = UD.astype(bool)
    Db  = (D >= thr_pixel)
    is_arid = (STb | Db).rename("is_arid").astype(bool)

    A = is_arid.fillna(False).astype(np.uint8).values
    p = uniform_filter(A.astype(float), size=k, mode="nearest")
    p_arid = xr.DataArray(p, coords=is_arid.coords, dims=is_arid.dims, name="p_arid")

    regional_desert = (is_arid & (p_arid >= tau_strict)).rename("regional_desert")
    mosaic          = (is_arid & (p_arid >= tau_mid) & (p_arid < tau_strict)).rename("mosaic")

    cls = xr.zeros_like(D, dtype="uint8").rename("desert_class2")
    cls = xr.where(mosaic, 3, cls)
    cls = xr.where(regional_desert, 1, cls)
    cls = xr.where(UDb, 2, cls)

    montane_mask = xr.zeros_like(D, dtype=bool).rename("montane_mask")
    if elev is not None:
        E = elev.rename("elev").where(np.isfinite)
        e = E.fillna(E.median()).values
        e_max = maximum_filter(e, size=k, mode="nearest")
        e_min = minimum_filter(e, size=k, mode="nearest")
        relief = xr.DataArray(e_max - e_min, coords=E.coords, dims=E.dims, name="relief")
        montane_mask = ((E >= montane_med_elev) | (relief >= montane_relief))
        montane_mask = (montane_mask & ((cls==1) | (cls==3))).rename("montane_mask")

    return {
        "is_arid": is_arid.astype("uint8"),
        "p_arid": p_arid,
        "regional_desert": regional_desert.astype("uint8"),
        "mosaic": mosaic.astype("uint8"),
        "desert_class2": cls,
        "montane_mask": montane_mask.astype("uint8"),
    }

if __name__ == "__main__":
    import argparse, pathlib
    ap = argparse.ArgumentParser(description="Neighborhood-consistency labelling (function wrapper)")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--thr_pixel", type=float, default=0.70)
    ap.add_argument("--tau_strict", type=float, default=0.70)
    ap.add_argument("--tau_mid", type=float, default=0.40)
    ap.add_argument("--elev", type=str, default="")
    ap.add_argument("--montane_med_elev", type=float, default=1500.0)
    ap.add_argument("--montane_relief", type=float, default=500.0)
    args = ap.parse_args()

    OUT = pathlib.Path(args.outdir)
    D  = xr.open_dataarray(OUT / "desertness_mean.nc")
    ST = xr.open_dataarray(OUT / "supertrue_mask.nc")
    UD = xr.open_dataarray(OUT / "undetermined_mask.nc")
    elev = xr.open_dataarray(args.elev) if args.elev else None

    res = compute_regional_labels(
        D, ST, UD,
        k=args.k, thr_pixel=args.thr_pixel,
        tau_strict=args.tau_strict, tau_mid=args.tau_mid,
        elev=elev, montane_med_elev=args.montane_med_elev, montane_relief=args.montane_relief
    )
    for name, da in res.items():
        da.to_netcdf(OUT / f"{name}.nc")
    cls = res["desert_class2"]
    tot = int(np.prod([s for _, s in cls.sizes.items()]))
    print("counts:", {int(v): int(((cls==v).sum()).item()) for v in (0,1,2,3)}, "total=", tot)
    print("Saved to", OUT)
