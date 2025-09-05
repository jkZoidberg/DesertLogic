from __future__ import annotations
import pathlib, time, traceback, sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

import numpy as np
import xarray as xr

try:
    from viz_quick import save_quick_png  
except Exception:
    save_quick_png = None

def _now() -> str:
    return time.strftime("%H:%M:%S")

def _fmt_secs(t: float) -> str:
    return f"{t:.2f}s"

def _summ(obj: Any, name: str = "") -> str:
    try:
        if isinstance(obj, xr.DataArray):
            shape = tuple(int(obj.sizes[d]) for d in obj.dims)
            vmin = float(obj.min().values) if obj.size else float("nan")
            vmax = float(obj.max().values) if obj.size else float("nan")
            mean = float(obj.mean().values) if obj.size else float("nan")
            return f"{name} DataArray dims={tuple(obj.dims)} shape={shape}, min={vmin:.3f}, max={vmax:.3f}, mean={mean:.3f}"
        elif isinstance(obj, xr.Dataset):
            vars_ = list(obj.data_vars)
            return f"{name} Dataset vars={vars_}, coords={list(obj.coords)}"
        elif isinstance(obj, np.ndarray):
            vmin = float(np.nanmin(obj)) if obj.size else float("nan")
            vmax = float(np.nanmax(obj)) if obj.size else float("nan")
            mean = float(np.nanmean(obj)) if obj.size else float("nan")
            return f"{name} ndarray shape={obj.shape}, min={vmin:.3f}, max={vmax:.3f}, mean={mean:.3f}"
        elif np.isscalar(obj):
            return f"{name} scalar={obj!r} ({type(obj).__name__})"
        else:
            s = str(obj)
            if len(s) > 80:
                s = s[:77] + "..."
            return f"{name} {type(obj).__name__}: {s}"
    except Exception as e:
        return f"{name} <summary-error> {e.__class__.__name__}: {e}"

@dataclass
class PipelineDebugger:
    outdir: pathlib.Path = field(default_factory=lambda: pathlib.Path("out"))
    cfg_path: pathlib.Path = field(default_factory=lambda: pathlib.Path("config.yml"))
    verbose: bool = True
    _opened: Dict[str, Any] = field(default_factory=dict)

    def run_all(self) -> None:
        print(f"[{_now()}] ▶ Start stepwise debug in: {self.outdir.resolve()}")
        t0 = time.time()

        self.step_00_env_check()
        self.step_01_load_config()
        self.step_02_scan_outputs()
        self.step_03_open_core_layers()
        self.step_04_summarize_core()
        self.step_05_try_quick_pngs()
        self.step_06_check_supervaluation_masks()
        self.step_07_optional_geotiff_probe()

        print(f"[{_now()}] ✔ Done. Total {_fmt_secs(time.time() - t0)}")

    def step_00_env_check(self) -> None:
        t = time.time()
        try:
            import xarray as _; import numpy as _  # noqa
            try:
                import rioxarray as _  # noqa
                riox = "OK"
            except Exception:
                riox = "missing"
            print(f"[{_now()}] step_00_env_check: xarray OK, numpy OK, rioxarray={riox}")
        except Exception:
            print(f"[{_now()}] step_00_env_check: ! base libs not ready")
            traceback.print_exc()
        finally:
            if self.verbose:
                print("    ", _fmt_secs(time.time() - t))

    def step_01_load_config(self) -> None:
        t = time.time()
        try:
            if self.cfg_path.exists():
                import yaml
                cfg = yaml.safe_load(self.cfg_path.read_text(encoding="utf-8"))
                self._opened["config"] = cfg
                print(f"[{_now()}] step_01_load_config: loaded config.yml")
            else:
                print(f"[{_now()}] step_01_load_config: config.yml not found, skip")
        except Exception:
            print(f"[{_now()}] step_01_load_config: ! failed to read config")
            traceback.print_exc()
        finally:
            if self.verbose:
                print("    ", _fmt_secs(time.time() - t))

    def step_02_scan_outputs(self) -> None:
        t = time.time()
        try:
            if not self.outdir.exists():
                print(f"[{_now()}] step_02_scan_outputs: out/ not found")
                return
            nc = sorted(self.outdir.glob("*.nc"))
            tif = sorted(self.outdir.glob("*.tif"))
            png = sorted(self.outdir.glob("*.png"))
            print(f"[{_now()}] step_02_scan_outputs: "
                  f"found {len(nc)} .nc, {len(tif)} .tif, {len(png)} .png")
            self._opened["files_nc"] = nc
            self._opened["files_tif"] = tif
            self._opened["files_png"] = png
        except Exception:
            print(f"[{_now()}] step_02_scan_outputs: ! scan failed")
            traceback.print_exc()
        finally:
            if self.verbose:
                print("    ", _fmt_secs(time.time() - t))

    def _open_da(self, path: pathlib.Path) -> Optional[xr.DataArray]:
        try:
            da = xr.open_dataarray(path.as_posix())
            return da
        except Exception:
            try:
                ds = xr.open_dataset(path.as_posix())
                var = next(iter(ds.data_vars))
                return ds[var]
            except Exception:
                return None

    def _open_if_exists(self, name: str) -> Optional[xr.DataArray]:
        p = self.outdir / name
        if p.exists():
            da = self._open_da(p)
            if da is not None:
                self._opened[name] = da
                print("   ", _summ(da, f"[open] {name}"))
            else:
                print(f"    [open] {name} failed to open")
            return da
        else:
            print(f"    [open] {name} not found")
            return None

    def step_03_open_core_layers(self) -> None:
        t = time.time()
        print(f"[{_now()}] step_03_open_core_layers:")
        targets = [
            "months_valid_ppt.nc", "months_valid_pet.nc", "months_valid_both.nc",
            "full_years_ppt.nc", "full_years_pet.nc", "full_years_both.nc",
            "desertness_mean.nc", "supertrue_mask.nc", "undetermined_mask.nc"
        ]
        for fname in targets:
            self._open_if_exists(fname)
        if self.verbose:
            print("    ", _fmt_secs(time.time() - t))

    def step_04_summarize_core(self) -> None:
        t = time.time()
        print(f"[{_now()}] step_04_summarize_core:")
        for key in ["desertness_mean.nc", "supertrue_mask.nc", "undetermined_mask.nc"]:
            da = self._opened.get(key)
            if isinstance(da, xr.DataArray):
                print("   ", _summ(da, f"[sum] {key}"))
                try:
                    if da.dtype.kind in ("b", "i", "u", "f"):
                        val = float(da.mean().values) * 100.0
                        print(f"     -> coverage={val:.2f}% (mean*100)")
                except Exception:
                    pass
        if self.verbose:
            print("    ", _fmt_secs(time.time() - t))

    def step_05_try_quick_pngs(self) -> None:
        t = time.time()
        print(f"[{_now()}] step_05_try_quick_pngs:")
        if save_quick_png is None:
            print("    viz_quick.save_quick_png not available, skip")
            if self.verbose:
                print("    ", _fmt_secs(time.time() - t))
            return
        for key in ["desertness_mean.nc", "supertrue_mask.nc", "undetermined_mask.nc"]:
            da = self._opened.get(key)
            if isinstance(da, xr.DataArray):
                out_png = (self.outdir / f"{pathlib.Path(key).stem}_quick.png").as_posix()
                try:
                    save_quick_png(da, out_png)
                    print(f"    wrote {out_png}")
                except Exception as e:
                    print(f"    quick png failed for {key}: {e}")
        if self.verbose:
            print("    ", _fmt_secs(time.time() - t))

    def step_06_check_supervaluation_masks(self) -> None:
        t = time.time()
        print(f"[{_now()}] step_06_check_supervaluation_masks:")
        st = self._opened.get("supertrue_mask.nc")
        ud = self._opened.get("undetermined_mask.nc")
        if isinstance(st, xr.DataArray) and isinstance(ud, xr.DataArray):
            try:
                same_shape = (tuple(st.sizes.values()) == tuple(ud.sizes.values()))
                print(f"    same_shape={same_shape}")
                st_bad = int(((st < 0) | (st > 1)).sum())
                ud_bad = int(((ud < 0) | (ud > 1)).sum())
                print(f"    value_range_violations: ST={st_bad}, UD={ud_bad}")
                overlap = float((xr.where((st > 0.5) & (ud > 0.5), 1, 0)).mean().values) * 100.0
                print(f"    overlap(>0.5)={overlap:.3f}%")
            except Exception:
                print("    mask checks failed")
                traceback.print_exc()
        else:
            print("    masks not found or failed to open")
        if self.verbose:
            print("    ", _fmt_secs(time.time() - t))

    def step_07_optional_geotiff_probe(self) -> None:
        t = time.time()
        print(f"[{_now()}] step_07_optional_geotiff_probe:")
        try:
            tifs: List[pathlib.Path] = self._opened.get("files_tif", [])
            if not tifs:
                print("    no GeoTIFF files, skip")
                return
            for p in tifs[:3]:  
                try:
                    da = self._open_da(p)
                    if isinstance(da, xr.DataArray):
                        print("   ", _summ(da, f"[tif] {p.name}"))
                    else:
                        print(f"    open {p.name} -> not a DataArray, skipped")
                except Exception as e:
                    print(f"    open {p.name} failed: {e}")
        except Exception:
            traceback.print_exc()
        finally:
            if self.verbose:
                print("    ", _fmt_secs(time.time() - t))


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("out")
    cfg = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("config.yml")
    dbg = PipelineDebugger(outdir=out, cfg_path=cfg, verbose=True)
    dbg.run_all()