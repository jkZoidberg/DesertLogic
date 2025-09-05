import argparse, pathlib, numpy as np, xarray as xr
from utils_terraclimate import fetch_terraclimate
from validators import ConfigValidator, ValidationError
import pathlib
def load_cfg(config_path: str = "config.yml", *, strict: bool = True):
    p = pathlib.Path(config_path)
    if not p.exists():
        if strict:
            raise FileNotFoundError(f"File not foiund：{p.resolve()}")
        return (-30.0, 40.0, -20.0, 90.0), ("2016-01", "2020-12")

    try:
        vc = ConfigValidator().validate_yaml(p)
        return vc.bbox, vc.period
    except (ValidationError, ValueError) as e:
        if strict:
            raise
        print(f"[warn] 配置无效，使用默认值：{e}")
        return (-30.0, 40.0, -20.0, 90.0), ("2016-01", "2020-12")

def summarize_months(da, months, label):
    total_cells = da.sizes["lat"] * da.sizes["lon"]
    arr = da.values
    n_all  = int((arr == months).sum())
    n_any  = int((arr <  months).sum())
    n_zero = int((arr == 0).sum())
    print(f"[{label}] all months OK={n_all} ({n_all/total_cells*100:.2f}%), "
          f"any missing={n_any} ({n_any/total_cells*100:.2f}%), zero months={n_zero}")

def summarize_years(da, years, label):
    total_cells = da.sizes["lat"] * da.sizes["lon"]
    arr = da.values
    n_full = int((arr == years).sum())
    n_part = int(((arr > 0) & (arr < years)).sum())
    n_zero = int((arr == 0).sum())
    print(f"[{label}] all years full={n_full} ({n_full/total_cells*100:.2f}%), "
          f"partial years={n_part} ({n_part/total_cells*100:.2f}%), zero full years={n_zero}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速抽样：跳格下采样以加速统计")
    ap.add_argument("--stride", type=int, default=6, help="quick 模式的步长（默认每 6 像元取 1 个）")
    ap.add_argument("--years-chunk", type=int, default=1, help="增量逐年处理的年块大小（默认 1 年）")
    args = ap.parse_args()

    bbox, period = load_cfg()
    print("Using bbox:", bbox, "period:", period, "| quick:", args.quick, "stride:", args.stride)

    # month data
    ppt = fetch_terraclimate("ppt", period, bbox, agg="none")
    pet = fetch_terraclimate("pet", period, bbox, agg="none")

    # quick sample
    if args.quick:
        ppt = ppt.isel(lat=slice(None, None, args.stride), lon=slice(None, None, args.stride))
        pet = pet.isel(lat=slice(None, None, args.stride), lon=slice(None, None, args.stride))

    months = ppt.sizes["time"]
    years  = int(np.unique(ppt["time"].dt.year.values).size)
    print(f"Months in range: {months}, Years: {years}")

    tmpl = ppt.isel(time=0, drop=True)
    mv_ppt  = xr.zeros_like(tmpl, dtype="int32").rename("months_valid_ppt")
    mv_pet  = xr.zeros_like(tmpl, dtype="int32").rename("months_valid_pet")
    mv_both = xr.zeros_like(tmpl, dtype="int32").rename("months_valid_both")
    fy_ppt  = xr.zeros_like(tmpl, dtype="int32").rename("full_years_ppt")
    fy_pet  = xr.zeros_like(tmpl, dtype="int32").rename("full_years_pet")
    fy_both = xr.zeros_like(tmpl, dtype="int32").rename("full_years_both")

    years_list = np.unique(ppt["time"].dt.year.values).tolist()

    for i in range(0, len(years_list), args.years_chunk):
        chunk_years = years_list[i:i+args.years_chunk]
        y0, y1 = chunk_years[0], chunk_years[-1]
        print(f"[{i+1}/{len(years_list)}] processing years {y0}-{y1} ...")

        # collect 12*chunk data
        ppt_y = ppt.sel(time=slice(f"{y0}-01", f"{y1}-12")).notnull()
        pet_y = pet.sel(time=slice(f"{y0}-01", f"{y1}-12")).notnull()
        both  = ppt_y & pet_y

        # calculate valid year
        mv_ppt  = mv_ppt  + ppt_y.sum("time").astype("int32").load()
        mv_pet  = mv_pet  + pet_y.sum("time").astype("int32").load()
        mv_both = mv_both + both.sum("time").astype("int32").load()

        full_ppt  = (ppt_y.groupby("time.year").sum("time") == 12).sum("year").astype("int32").load()
        full_pet  = (pet_y.groupby("time.year").sum("time") == 12).sum("year").astype("int32").load()
        full_both = (both.groupby("time.year").sum("time") == 12).sum("year").astype("int32").load()
        fy_ppt  = fy_ppt  + full_ppt
        fy_pet  = fy_pet  + full_pet
        fy_both = fy_both + full_both

    print("\n--- Monthly coverage ---")
    summarize_months(mv_ppt, months,  "ppt")
    summarize_months(mv_pet, months,  "pet")
    summarize_months(mv_both, months, "both(ppt&pet)")

    print("\n--- Full-year coverage (12 valid months per year) ---")
    summarize_years(fy_ppt, years,  "ppt")
    summarize_years(fy_pet, years,  "pet")
    summarize_years(fy_both, years, "both(ppt&pet)")

    out = pathlib.Path("out"); out.mkdir(exist_ok=True)
    mv_ppt.to_netcdf(out/"months_valid_ppt.nc")
    mv_pet.to_netcdf(out/"months_valid_pet.nc")
    mv_both.to_netcdf(out/"months_valid_both.nc")
    fy_ppt.to_netcdf(out/"full_years_ppt.nc")
    fy_pet.to_netcdf(out/"full_years_pet.nc")
    fy_both.to_netcdf(out/"full_years_both.nc")
    print("\nSaved per-cell coverage rasters to 'out/'")

def run_coverage(
    quick: bool = False,
    stride: int = 6,
    years_chunk: int = 1,
    config_path: str = "config.yml",
):
    bbox, period = load_cfg(config_path)
    print("Using bbox:", bbox, "period:", period, "| quick:", quick, "stride:", stride)

    # month data
    ppt = fetch_terraclimate("ppt", period, bbox, agg="none")
    pet = fetch_terraclimate("pet", period, bbox, agg="none")

    # quick sample
    if quick:
        ppt = ppt.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
        pet = pet.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))

    months = ppt.sizes["time"]
    years  = int(np.unique(ppt["time"].dt.year.values).size)
    print(f"Months in range: {months}, Years: {years}")

    tmpl = ppt.isel(time=0, drop=True)
    mv_ppt  = xr.zeros_like(tmpl, dtype="int32").rename("months_valid_ppt")
    mv_pet  = xr.zeros_like(tmpl, dtype="int32").rename("months_valid_pet")
    mv_both = xr.zeros_like(tmpl, dtype="int32").rename("months_valid_both")
    fy_ppt  = xr.zeros_like(tmpl, dtype="int32").rename("full_years_ppt")
    fy_pet  = xr.zeros_like(tmpl, dtype="int32").rename("full_years_pet")
    fy_both = xr.zeros_like(tmpl, dtype="int32").rename("full_years_both")

    years_list = np.unique(ppt["time"].dt.year.values).tolist()

    for i in range(0, len(years_list), years_chunk):
        chunk_years = years_list[i:i+years_chunk]
        y0, y1 = chunk_years[0], chunk_years[-1]
        print(f"[{i+1}/{len(years_list)}] processing years {y0}-{y1} ...")

        # collect 12*chunk data
        ppt_y = ppt.sel(time=slice(f"{y0}-01", f"{y1}-12")).notnull()
        pet_y = pet.sel(time=slice(f"{y0}-01", f"{y1}-12")).notnull()
        both  = ppt_y & pet_y

        # calculate valid months
        mv_ppt  = mv_ppt  + ppt_y.sum("time").astype("int32").load()
        mv_pet  = mv_pet  + pet_y.sum("time").astype("int32").load()
        mv_both = mv_both + both.sum("time").astype("int32").load()

        # calculate full years (12 valid months per year)
        full_ppt  = (ppt_y.groupby("time.year").sum("time") == 12).sum("year").astype("int32").load()
        full_pet  = (pet_y.groupby("time.year").sum("time") == 12).sum("year").astype("int32").load()
        full_both = (both.groupby("time.year").sum("time") == 12).sum("year").astype("int32").load()
        fy_ppt  = fy_ppt  + full_ppt
        fy_pet  = fy_pet  + full_pet
        fy_both = fy_both + full_both

    print("\n--- Monthly coverage ---")
    summarize_months(mv_ppt, months,  "ppt")
    summarize_months(mv_pet, months,  "pet")
    summarize_months(mv_both, months, "both(ppt&pet)")

    print("\n--- Full-year coverage (12 valid months per year) ---")
    summarize_years(fy_ppt, years,  "ppt")
    summarize_years(fy_pet, years,  "pet")
    summarize_years(fy_both, years, "both(ppt&pet)")

    out = pathlib.Path("out"); out.mkdir(exist_ok=True)
    mv_ppt.to_netcdf(out/"months_valid_ppt.nc")
    mv_pet.to_netcdf(out/"months_valid_pet.nc")
    mv_both.to_netcdf(out/"months_valid_both.nc")
    fy_ppt.to_netcdf(out/"full_years_ppt.nc")
    fy_pet.to_netcdf(out/"full_years_pet.nc")
    fy_both.to_netcdf(out/"full_years_both.nc")
    print("\nSaved per-cell coverage rasters to 'out/'")

    return {
        "months_valid_ppt": mv_ppt,
        "months_valid_pet": mv_pet,
        "months_valid_both": mv_both,
        "full_years_ppt": fy_ppt,
        "full_years_pet": fy_pet,
        "full_years_both": fy_both,
        "months": months,
        "years": years,
        "bbox": bbox,
        "period": period,
    }
run_coverage