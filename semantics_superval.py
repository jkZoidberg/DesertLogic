import numpy as np
import xarray as xr

def _s_alpha_thresh(a: float, b: float, alpha: float) -> float:
    # s(x;a,b) >= alpha → x >= t   
    if alpha == 0.5:
        return (a + b) / 2.0
    return a + (b - a) * (1 - np.arccos(2*alpha - 1) / np.pi)

def _z_alpha_thresh(a: float, b: float, alpha: float) -> float:
    # z(x;a,b) >= alpha → x <= t
    if alpha == 0.5:
        return (a + b) / 2.0
    return a + (b - a) * (np.arccos(2*alpha - 1) / np.pi)

def superval_masks_from_precisifications(phi: xr.DataArray,
                                         ai: xr.DataArray,
                                         cwd: xr.DataArray | None,
                                         sv_cfg: dict,
                                         valid_mask: xr.DataArray | None = None):
    prec  = sv_cfg.get("precisifications", {})
    alpha = float(sv_cfg.get("alpha", 0.5))
    phi_s = prec.get("phi_s", [])
    ai_z  = prec.get("ai_z",  [])
    cwd_s = prec.get("cwd_s", [])

    if not (phi_s and ai_z):
        raise ValueError("supervaluation.precisifications need phi_s[] and ai_z[]")

    masks = []
    use_cwd = bool(cwd_s) and (cwd is not None)

    for ps in phi_s:
        tau_phi = _s_alpha_thresh(ps["a"], ps["b"], alpha)
        for az in ai_z:
            tau_ai = _z_alpha_thresh(az["a"], az["b"], alpha)
            if use_cwd:
                for cz in cwd_s:
                    tau_cwd = _s_alpha_thresh(cz["a"], cz["b"], alpha)
                    conds = [phi >= tau_phi, ai <= tau_ai, cwd >= tau_cwd]
                    m = xr.concat(conds, "cond").all("cond")
                    masks.append(m)
            else:
                m = xr.concat([phi >= tau_phi, ai <= tau_ai], "cond").all("cond")
                masks.append(m)

    M = xr.concat(masks, dim="theta")

    if valid_mask is None:
        valid_mask = xr.ufuncs.isfinite(phi) & xr.ufuncs.isfinite(ai)
        if use_cwd:
            valid_mask = valid_mask & xr.ufuncs.isfinite(cwd)

    M = xr.where(valid_mask, M, False)

    ST = M.all("theta").rename("supertrue_mask")
    SF = (~M).all("theta").rename("superfalse_mask")
    UD = (~ST) & (~SF)
    votes_true  = M.sum("theta").rename("votes_true")
    votes_total = xr.full_like(votes_true, M.sizes["theta"]).rename("votes_total")
    return ST, SF, UD, votes_true, votes_total