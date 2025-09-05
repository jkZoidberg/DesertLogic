import xarray as xr

def _vote_fraction(votes_true: xr.DataArray, votes_total: xr.DataArray) -> xr.DataArray:
    """safe votes_true / votes_total, handling zeros and NaNs."""
    frac = xr.where((votes_total > 0) & votes_total.notnull(),
                    votes_true / votes_total, 0.0)
    return frac.rename("votes_true_fraction")

def rule_ud_to_stweak(UD: xr.DataArray,
                      votes_true: xr.DataArray,
                      votes_total: xr.DataArray,
                      thr: float = 0.65) -> xr.DataArray:
    frac = _vote_fraction(votes_true, votes_total)
    mask = (UD.astype(bool) & (frac >= float(thr))).rename("rdr_ST_weak_mask")
    return mask

def rule_ud_to_sfweak(UD: xr.DataArray,
                      votes_true: xr.DataArray,
                      votes_total: xr.DataArray,
                      thr: float = 0.34) -> xr.DataArray:
    frac = _vote_fraction(votes_true, votes_total)
    mask = (UD.astype(bool) & (frac <= float(thr))).rename("rdr_SF_weak_mask")
    return mask

def rule_ud_to_st_neighborhood(UD: xr.DataArray, p_arid: xr.DataArray, tau_strict: float = 0.70):
    return (UD.astype(bool) & (p_arid >= float(tau_strict))).rename("rdr_ST_strong_neigh")

def rule_ud_to_sf_neighborhood(UD: xr.DataArray, p_arid: xr.DataArray, tau_mid: float = 0.40):
    return (UD.astype(bool) & (p_arid < float(tau_mid))).rename("rdr_SF_strong_neigh")