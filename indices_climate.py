# indices_climate.py

import numpy as np
import xarray as xr

_EPS = 1e-9  # avoid devided by 0

def annual_sum(da_monthly: xr.DataArray) -> xr.DataArray:
    
    return da_monthly.groupby("time.year").sum("time", keep_attrs=True)

def climate_mean(da_yearly: xr.DataArray) -> xr.DataArray:
    
    return da_yearly.mean("year", keep_attrs=True)

def dryness_ratio_phi(pet_clim: xr.DataArray, ppt_clim: xr.DataArray) -> xr.DataArray:
    
    return (pet_clim / (ppt_clim + _EPS)).rename("phi_PET_over_P")

def aridity_index_ai(ppt_clim: xr.DataArray, pet_clim: xr.DataArray) -> xr.DataArray:
    
    return (ppt_clim / (pet_clim + _EPS)).rename("AI_P_over_PET")



def evaporative_stress_index(aet_clim: xr.DataArray, pet_clim: xr.DataArray) -> xr.DataArray:
    return (aet_clim / (pet_clim + _EPS)).where(aet_clim.notnull() & pet_clim.notnull()).rename("ESI_AET_over_PET")

def climatic_water_deficit_from_def(def_yearly: xr.DataArray) -> xr.DataArray:
    return def_yearly.mean("year", skipna=True).rename("CWD_mm_per_year")

def climatic_water_deficit_from_pet_aet(pet_clim: xr.DataArray, aet_clim: xr.DataArray) -> xr.DataArray:
    return (pet_clim - aet_clim).rename("CWD_mm_per_year")

def thornthwaite_moisture_index(p_clim: xr.DataArray, pet_clim: xr.DataArray) -> xr.DataArray:
    return (100.0 * (p_clim - pet_clim) / (pet_clim + _EPS)).where(p_clim.notnull() & pet_clim.notnull()).rename("TMI_percent")

def tmean_monthly(tmax_m: xr.DataArray, tmin_m: xr.DataArray) -> xr.DataArray:
    tmax_m, tmin_m = xr.align(tmax_m, tmin_m)
    return ((tmax_m + tmin_m) / 2.0).rename("tmean")

def dry_month_count(ppt_m: xr.DataArray, tmax_m: xr.DataArray, tmin_m: xr.DataArray) -> xr.DataArray:
    Tm = tmean_monthly(tmax_m, tmin_m)
    dry = ppt_m < (2.0 * Tm)                         # Walter 判据
    dry_y = dry.groupby("time.year").sum("time")
    return dry_y.mean("year").rename("DryMonths_per_year")

def seasonality_index(ppt_m: xr.DataArray) -> xr.DataArray:
    def _si_one_year(p_m):
        Ptot = p_m.sum("time")
        Pbar = Ptot / 12.0
        return (abs(p_m - Pbar)).sum("time") / (Ptot + _EPS)
    si_y = ppt_m.groupby("time.year").apply(_si_one_year)
    return si_y.mean("year").rename("SeasonalityIndex")

def vpd_climatology(vpd_m: xr.DataArray) -> xr.DataArray:
    return vpd_m.groupby("time.year").mean("time").mean("year").rename("VPD_kPa")

def runoff_ratio(runoff_clim: xr.DataArray, p_clim: xr.DataArray) -> xr.DataArray:
    return (runoff_clim / (p_clim + _EPS)).where(runoff_clim.notnull() & p_clim.notnull()).rename("RunoffRatio")

def de_martonne(p_clim: xr.DataArray, tmean_clim: xr.DataArray) -> xr.DataArray:
    return (p_clim / (tmean_clim + 10.0)).rename("IDM_DeMartonne")
