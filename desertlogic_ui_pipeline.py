from __future__ import annotations

import pathlib
import re
import traceback
from typing import Dict, Optional

import numpy as np
import streamlit as st
import xarray as xr
import yaml


try:
    import pipeline as pl  # expects: step1_fetch_and_indices, step2_fuzzy, step3_supervaluation, run_desertness
except Exception as e:
    pl = None
    _pip_err = e
else:
    _pip_err = None

# Optional: coverage & RDR
try:
    from coverage_report import run_coverage
except Exception as e:
    run_coverage = None
    _cov_err = e
else:
    _cov_err = None

try:
    from regional_consistency import compute_regional_labels
except Exception as e:
    compute_regional_labels = None
    _rc_err = e
else:
    _rc_err = None

try:
    from rdr_apply import run_phase1 as rdr_phase1, run_phase2 as rdr_phase2, finalize_labels as rdr_finalize
except Exception as e:
    rdr_phase1 = rdr_phase2 = rdr_finalize = None
    _rdr_err = e
else:
    _rdr_err = None

# ----------------------------- Small helpers -----------------------------

def skey(*parts):
    """Stable, readable Streamlit widget key generator."""
    s = "_".join(str(p) for p in parts if p is not None)
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def _plot_da(da: xr.DataArray, title: str):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    da.plot(ax=ax, add_colorbar=True)
    ax.set_title(title)
    st.pyplot(fig)


def _summ(da) -> str:
    import numpy as np
    import xarray as xr
    try:
        if isinstance(da, xr.DataArray):
            vals = da.values  # numpy array
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            vmean = float(np.nanmean(vals))
            return f"shape={tuple(da.sizes.values())}, min={vmin:.3f}, max={vmax:.3f}, mean={vmean:.3f}"
        # Non-DataArray (int/float/str/None): show type & value
        return f"type={type(da).__name__}, value={da!r}"
    except Exception as e:
        # Fallback: show type + brief error
        return f"type={type(da).__name__}, error={e}"


def _read_cfg_text(path: pathlib.Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    # default template (safe, minimal, and valid for the pipeline)
    return (
        "region:\n"
        "  bbox: [0.0, 28.0, -10.0, 10.0]\n"
        "  period: [\"2018-01\", \"2020-12\"]\n"
        "output:\n"
        "  dir: out\n"
        "fuzzy:\n"
        "  phi_s: {a: 10.0, b: 30.0}\n"
        "  ai_z:  {a: 0.02, b: 0.15}\n"
        "  cwd_s: {a: 2400.0, b: 3000.0}\n"
        "  combine: \"np.maximum.reduce([mu_phi, 1-mu_ai, mu_cwd])\"\n"
        "supervaluation:\n"
        "  thresholds: {supertrue: 0.80, superfalse: 0.20}\n"
        "  alpha: 0.5\n"
        "  precisifications:\n"
        "    phi_s:\n"
        "      - {a: 12.0, b: 28.0}\n"
        "      - {a: 15.0, b: 35.0}\n"
        "      - {a: 10.0, b: 30.0}\n"
        "    ai_z:\n"
        "      - {a: 0.025, b: 0.13}\n"
        "      - {a: 0.030, b: 0.16}\n"
        "      - {a: 0.020, b: 0.15}\n"
        "    cwd_s:\n"
        "      - {a: 2400.0, b: 2900.0}\n"
        "      - {a: 2500.0, b: 3000.0}\n"
        "      - {a: 2600.0, b: 3100.0}\n"
    )


def _validate_cfg_text(text: str) -> Optional[str]:
    try:
        from validators import ConfigValidator
        cfg = yaml.safe_load(text) or {}
        ConfigValidator().validate_dict(cfg)
        return None
    except Exception as e:
        return f"Config validation failed: {e}"


def _load_cfg(path: pathlib.Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

# ----------------------------- UI layout -----------------------------

st.set_page_config(page_title="DesertLogic UI (pipeline) - Stepwise", layout="wide")
st.title("DesertLogic UI - Stepwise visualization via pipeline.py + Config editor")

# Sidebar: config editor
with st.sidebar:
    st.header("config.yml editor")
    cfg_path_str = st.text_input("Config file path", value=st.session_state.get("cfg_path", "config.yml"), key=skey("inp", "cfg", "path"))
    cfg_path = pathlib.Path(cfg_path_str)
    if "cfg_text" not in st.session_state or st.session_state.get("cfg_path") != cfg_path_str:
        st.session_state["cfg_text"] = _read_cfg_text(cfg_path)
        st.session_state["cfg_path"] = cfg_path_str

    cols = st.columns(3)
    if cols[0].button("Reload", use_container_width=True, key=skey("btn", "cfg", "reload")):
        st.session_state["cfg_text"] = _read_cfg_text(cfg_path)
        st.write("Reloaded from disk.")
    if cols[1].button("Validate", use_container_width=True, key=skey("btn", "cfg", "validate")):
        err = _validate_cfg_text(st.session_state["cfg_text"])
        st.write("✓ Config validated.") if err is None else st.write(err)
    if cols[2].button("Save", use_container_width=True, key=skey("btn", "cfg", "save")):
        err = _validate_cfg_text(st.session_state["cfg_text"])    
        if err is None:
            cfg_path.write_text(st.session_state["cfg_text"], encoding="utf-8")
            st.write(f"Saved: {cfg_path.resolve()}")
        else:
            st.write(err)

    st.text_area("YAML", key="cfg_text", height=260, label_visibility="collapsed")

    # Optional: clear session-state cache to avoid stale objects
    if st.button("Clear session cache", key=skey("btn", "cache", "clear")):
        for k in list(st.session_state.keys()):
            if k not in ("cfg_text", "cfg_path", skey("inp", "cfg", "path")):
                del st.session_state[k]
        st.write("Cleared.")

# Tabs
T_cov, T_s1, T_s2, T_s3, T_rdr = st.tabs(["Coverage (optional)", "Step 1 - Indices", "Step 2 - Fuzzy", "Step 3 - Supervaluation", "Step 4 - RDR"])

# ----------------------------- Coverage (optional) -----------------------------
with T_cov:
    st.subheader("Optional: Coverage stats (ppt & pet)")
    if run_coverage is None:
        msg = "Could not import coverage_report.run_coverage; you may skip this tab."
        if _cov_err:
            msg += "".join(traceback.format_exception_only(type(_cov_err), _cov_err))
        st.write(msg)
    else:
        c1, c2 = st.columns(2)
        if c1.button("Run coverage", key=skey("btn", "cov", "run")):
            try:
                outs = run_coverage(quick=True, stride=6, years_chunk=1, config_path=str(cfg_path))
                st.session_state["cov_outs"] = outs
                st.write("Coverage results saved to out/")
            except Exception as e:
                st.exception(e)
        if c2.button("Load from out/", key=skey("btn", "cov", "load")):
            out = pathlib.Path("out")
            keys = [
                "months_valid_ppt.nc","months_valid_pet.nc","months_valid_both.nc",
                "full_years_ppt.nc","full_years_pet.nc","full_years_both.nc",
            ]
            loaded = {}
            for k in keys:
                p = out/k
                if p.exists():
                    loaded[p.stem] = xr.open_dataarray(p)
            if loaded:
                st.session_state["cov_outs"] = loaded
                st.write("Loaded coverage artifacts from out/")
            else:
                st.write("No coverage files found")
        outs = st.session_state.get("cov_outs")
        if outs:
            cols = st.columns(3)
            for i, (name, da) in enumerate(outs.items()):
                with cols[i % 3]:
                    st.caption(f"**{name}** - {_summ(da)}")
                    if st.button(f"Show {name}", key=skey("btn", "cov", "show", name)):
                        _plot_da(da, name)

# ----------------------------- Step 1: indices -----------------------------
with T_s1:
    st.subheader("Step 1 - Fetch & indices (P, PET, phi, AI, CWD)")
    if pl is None:
        st.write("Failed to import pipeline.py" + "".join(traceback.format_exception_only(type(_pip_err), _pip_err)))
    else:
        c1, c2 = st.columns(2)
        if c1.button("Run Step 1", key=skey("btn", "s1", "run")):
            try:
                cfg = _load_cfg(cfg_path)
                s1 = pl.step1_fetch_and_indices(cfg, write_nc=True)
                st.session_state["s1"] = s1
                st.write("Step 1 done: P/PET/phi/AI/CWD written to out/ and cached.")
            except Exception as e:
                st.exception(e)
        if c2.button("Load from out/", key=skey("btn", "s1", "load")):
            try:
                out = pathlib.Path("out")
                s1 = {}
                for name in ["P","PET","phi","ai","CWD"]:
                    p = out/f"{name}.nc"
                    if p.exists():
                        s1[name] = xr.open_dataarray(p)
                if s1:
                    st.session_state["s1"] = s1
                    st.write("Loaded indices from out/")
                else:
                    st.write("Not found P/PET/phi/ai/CWD")
            except Exception as e:
                st.exception(e)

        s1 = st.session_state.get("s1")
        if s1:
            cols = st.columns(3)
            for i, k in enumerate(["P","PET","phi","ai","CWD"]):
                da = s1.get(k)
                if isinstance(da, xr.DataArray):
                    with cols[i % 3]:
                        st.caption(f"**{k}** - {_summ(da)}")
                        if st.button(f"Show {k}", key=skey("btn", "s1", "show", k)):
                            _plot_da(da, k)

# ----------------------------- Step 2: fuzzy -----------------------------
with T_s2:
    st.subheader("Step 2 - Fuzzy combination (desertness_mean)")
    if pl is None:
        st.write("pipeline not imported; cannot run this step.")
    else:
        if st.button("Run Step 2 (Fuzzy only)", key=skey("btn", "s2", "run")):
            try:
                cfg = _load_cfg(cfg_path)
                s1 = st.session_state.get("s1") or {
                    k: xr.open_dataarray(f"out/{k}.nc") for k in ["phi","ai","CWD"]
                }
                D = pl.step2_fuzzy(cfg, phi=s1["phi"], ai=s1["ai"], CWD=s1["CWD"], write_nc=True, export_geotiff=False)
                st.session_state["desertness_mean"] = D
                st.write("desertness_mean generated and written to out/")
            except Exception as e:
                st.exception(e)
        D = st.session_state.get("desertness_mean")
        if not isinstance(D, xr.DataArray):
            p = pathlib.Path("out/desertness_mean.nc")
            D = xr.load_dataarray(p) if p.exists() else None
    if isinstance(D, xr.DataArray):
        st.caption(f"**desertness_mean** - {_summ(D)}")
    if st.button("Show desertness_mean", key=skey("btn","s2","show","D")):
        _plot_da(D, "desertness_mean")
# ----------------------------- Step 3: supervaluation -----------------------------
with T_s3:
    st.subheader("Step 3 - Supervaluation (ST/SF/UD + votes)")
    if pl is None:
        st.info("pipeline 未导入，无法运行此步。")
    else:
        if st.button("Run Step 3 (SV)", key=skey("btn", "s3", "run")):
            try:
                cfg = _load_cfg(cfg_path)
                phi = st.session_state.get("s1", {}).get("phi")
                if not isinstance(phi, xr.DataArray):
                    phi = xr.load_dataarray("out/phi.nc")

                ai = st.session_state.get("s1", {}).get("ai")
                if not isinstance(ai, xr.DataArray):
                    ai = xr.load_dataarray("out/ai.nc")

                CWD = st.session_state.get("s1", {}).get("CWD")
                if not isinstance(CWD, xr.DataArray):
                    CWD = xr.load_dataarray("out/CWD.nc")

                D = st.session_state.get("desertness_mean")
                if not isinstance(D, xr.DataArray):
                    D = xr.load_dataarray("out/desertness_mean.nc")
                sv = pl.step3_supervaluation(cfg, phi=phi, ai=ai, CWD=CWD, desertness=D, write_nc=True, export_geotiff=False)
                st.session_state["sv"] = sv
                st.write("SV done: ST/SF/UD (+votes) written to out/")
            except Exception as e:
                st.exception(e)

        sv = st.session_state.get("sv")
        names = [
            ("supertrue_mask", "ST (supertrue)"),
            ("undetermined_mask", "UD (undetermined)"),
            ("superfalse_mask", "SF (superfalse)"),
        ]
        cols = st.columns(3)
        for i, (k, label) in enumerate(names):
            p = pathlib.Path(f"out/{k}.nc")
            if p.exists():
                da = xr.open_dataarray(p)
                with cols[i % 3]:
                    st.caption(f"**{label}** - {_summ(da)}")
                    if st.button(f"Show {label}", key=skey("btn", "s3", "show", k)):
                        _plot_da(da, label)
        # show votes stats if present
        vt, vtt = pathlib.Path("out/votes_true.nc"), pathlib.Path("out/votes_total.nc")
        if vt.exists() and vtt.exists():
            VT, VTT = xr.open_dataarray(vt), xr.open_dataarray(vtt)
            frac = (VT / xr.where(VTT==0, np.nan, VTT)).mean().item()
            st.caption(f"Mean vote fraction (VT/VTT) ~ {frac:.3f}")

# ----------------------------- Step 4: RDR -----------------------------
with T_rdr:
    st.subheader("Step 4 - RDR (Phase1/2 & Final)")
    if _rdr_err:
        msg = "Could not import rdr_apply; if you don't need RDR, you can ignore this tab."
        msg += "".join(traceback.format_exception_only(type(_rdr_err), _rdr_err))
        st.write(msg)
    else:
        c1, c2, c3 = st.columns(3)
        st_thr = c1.number_input("ST weak threshold (votes_true fraction)", value=0.75, min_value=0.0, max_value=1.0, step=0.05, key=skey("num", "rdr", "st_thr"))
        sf_thr = c2.number_input("SF weak threshold (votes_true fraction)", value=0.25, min_value=0.0, max_value=1.0, step=0.05, key=skey("num", "rdr", "sf_thr"))
        compute_p = c3.button("Compute p_arid (neighborhood consistency)", key=skey("btn", "rdr", "compute_p"))

        if compute_p:
            if compute_regional_labels is None:
                st.write("regional_consistency not imported; cannot compute p_arid")
            else:
                try:
                    D  = xr.open_dataarray("out/desertness_mean.nc")
                    ST_da = xr.open_dataarray("out/supertrue_mask.nc")
                    UD_da = xr.open_dataarray("out/undetermined_mask.nc")
                    res = compute_regional_labels(D, ST_da, UD_da, k=11, thr_pixel=0.70, tau_strict=0.70, tau_mid=0.40)
                    res["p_arid"].to_netcdf("out/p_arid.nc")
                    st.write("p_arid.nc generated")
                except Exception as e:
                    st.exception(e)

        r1, r2, rf = st.columns(3)
        if r1.button("Run Phase 1 (weak rules)", key=skey("btn", "rdr", "phase1")):
            try:
                rdr_phase1(outdir="out", st_thr=float(st_thr), sf_thr=float(sf_thr), export_geotiff=False)
                st.write("Phase 1 done")
            except Exception as e:
                st.exception(e)
        if r2.button("Run Phase 2 (neighborhood)", key=skey("btn", "rdr", "phase2")):
            try:
                rdr_phase2(outdir="out", tau_strict=0.70, tau_mid=0.40, export_geotiff=False)
                st.write("Phase 2 done")
            except Exception as e:
                st.exception(e)
        if rf.button("Finalize labels", key=skey("btn", "rdr", "finalize")):
            try:
                rdr_finalize(outdir="out")
                st.write("desert_class_final.nc generated")
            except Exception as e:
                st.exception(e)

        # quick viewers
        show_list = [
            ("out/rdr_ST_weak_mask.nc", "rdr_ST_weak_mask"),
            ("out/rdr_SF_weak_mask.nc", "rdr_SF_weak_mask"),
            ("out/rdr_ST_strong_neigh.nc", "rdr_ST_strong_neigh"),
            ("out/rdr_SF_strong_neigh.nc", "rdr_SF_strong_neigh"),
            ("out/p_arid.nc", "p_arid"),
            ("out/desert_class_final.nc", "desert_class_final"),
        ]
        cols = st.columns(3)
        for i, (p, name) in enumerate(show_list):
            if pathlib.Path(p).exists():
                try:
                    da = xr.open_dataarray(p)
                    with cols[i % 3]:
                        st.caption(f"**{name}** - {_summ(da)}")
                        if st.button(f"Show {name}", key=skey("btn", "rdr", "show", name)):
                            _plot_da(da, name)
                except Exception:
                    pass
