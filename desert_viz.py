from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import xarray as xr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mpl_colors

try:
    import folium  
    _HAVE_FOLIUM = True
except Exception:
    _HAVE_FOLIUM = False


@dataclass
class VizConfig:
    cmap: str = "Spectral_r"     
    alpha_cont: float = 0.85      
    alpha_mask: float = 0.45      
    st_color: Tuple[int,int,int] = (0, 180, 0)   
    ud_color: Tuple[int,int,int] = (255, 140, 0)  
    sf_color: Tuple[int,int,int] = (128, 0, 153)  


class DesertViz:
    def __init__(self, cfg: Optional[VizConfig] = None):
        self.cfg = cfg or VizConfig()

    @staticmethod
    def _dims(da: xr.DataArray) -> Tuple[str, str]:
        ydim, xdim = da.dims[-2], da.dims[-1]
        return ydim, xdim

    @staticmethod
    def _bounds_from_coords(da: xr.DataArray) -> Tuple[float,float,float,float,bool]:
        lat_name, lon_name = da.dims[-2], da.dims[-1]
        lat = da[lat_name].values
        lon = da[lon_name].values
        top, bottom = float(np.nanmax(lat)), float(np.nanmin(lat))
        left, right = float(np.nanmin(lon)), float(np.nanmax(lon))
        flipud = lat[0] < lat[-1] 
        return (left, bottom, right, top, flipud)

    @staticmethod
    def _to_png_rgba(arr: np.ndarray, cmap_name: str, vmin=None, vmax=None, alpha=1.0) -> Image.Image:
        vmin = np.nanmin(arr) if vmin is None else vmin
        vmax = np.nanmax(arr) if vmax is None else vmax
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        rgba = cm.get_cmap(cmap_name)(norm(arr))
        rgba[..., -1] = np.where(np.isnan(arr), 0, alpha)
        return Image.fromarray((rgba*255).astype(np.uint8))

    @staticmethod
    def _mask_png(mask: np.ndarray, color: Tuple[int,int,int], alpha: float) -> Image.Image:
        h, w = mask.shape
        img = np.zeros((h,w,4), dtype=np.uint8)
        img[mask==1] = [*color, int(alpha*255)]
        return Image.fromarray(img, 'RGBA')

    def save_continuous_png(self, da: xr.DataArray, out_png: str, *, vmin=None, vmax=None):
        """把任意 DataArray（如 desertness_mean, p_arid 等）渲染为 PNG。"""
        left, bottom, right, top, flipud = self._bounds_from_coords(da)
        arr = da.values
        if flipud: arr = np.flipud(arr)
        img = self._to_png_rgba(arr, self.cfg.cmap, vmin=vmin, vmax=vmax, alpha=self.cfg.alpha_cont)
        img.save(out_png)
        return (left, bottom, right, top)

    def save_hist_png(self, data: np.ndarray, out_png: str, *, bins=40, rng=(0,1), title: str = "Histogram"):
        plt.figure(figsize=(6,4))
        plt.hist(data[np.isfinite(data)], bins=bins, range=rng, edgecolor="black", linewidth=0.5)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()

    def save_superval_overlay_png(
        self,
        base: xr.DataArray,  
        st: xr.DataArray,
        ud: xr.DataArray,
        sf: xr.DataArray,
        out_png: str,
        *, vmin=0.0, vmax=1.0, title: str = "Desertness + ST/UD/SF"
    ):
        valid = base.notnull()
        base = base.where(valid)
        st, ud, sf = st.astype(bool).where(valid, False), ud.astype(bool).where(valid, False), sf.astype(bool).where(valid, False)

        
        plt.figure(figsize=(8,5))
        ax = plt.gca()
        base.plot(ax=ax, cmap=self.cfg.cmap, add_colorbar=True, vmin=vmin, vmax=vmax)

        ydim, xdim = self._dims(base)
        H, W = base.sizes[ydim], base.sizes[xdim]
        rgba = np.zeros((H, W, 4), dtype=float)
        st_np = st.transpose(ydim, xdim).values
        ud_np = ud.transpose(ydim, xdim).values
        sf_np = sf.transpose(ydim, xdim).values
        rgba[st_np, :3] = np.array(self.cfg.st_color)/255.0
        rgba[ud_np, :3] = np.array(self.cfg.ud_color)/255.0
        rgba[sf_np, :3] = np.array(self.cfg.sf_color)/255.0
        rgba[(st_np | ud_np | sf_np), 3] = 0.35

        lat = base[ydim].values
        lon = base[xdim].values
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
        origin = "lower" if (lat[1] > lat[0]) else "upper"
        ax.imshow(rgba, extent=extent, origin=origin)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=170)
        plt.close()

    def save_interactive_map(
        self,
        base: xr.DataArray,             
        class_mask: Optional[xr.DataArray],  
        out_html: str = "map_desert.html",
        *, vmin=0.0, vmax=1.0
    ):
        if not _HAVE_FOLIUM:
            raise RuntimeError("folium 未安装：pip install folium")

        left, bottom, right, top, flipud = self._bounds_from_coords(base)
        arr = base.values
        if flipud: arr = np.flipud(arr)

       
        cont_png = out_html.replace('.html', '_cont.png')
        Image.fromarray((cm.get_cmap(self.cfg.cmap)(mpl_colors.Normalize(vmin=vmin, vmax=vmax)(arr))*255).astype(np.uint8)).save(cont_png)

        m = folium.Map(location=[(top+bottom)/2, (left+right)/2], zoom_start=5, tiles='CartoDB positron')
        folium.raster_layers.ImageOverlay(
            name='Desertness', image=cont_png,
            bounds=[[bottom, left], [top, right]], opacity=1.0, zindex=2).add_to(m)

        
        if class_mask is not None:
            cmask = (class_mask == 1).astype(np.uint8)
            m_arr = cmask.values
            if flipud: m_arr = np.flipud(m_arr)
            mask_png = out_html.replace('.html', '_mask.png')
            self._mask_png(m_arr, color=(255,0,0), alpha=self.cfg.alpha_mask).save(mask_png)
            folium.raster_layers.ImageOverlay(
                name='Desert mask (class=1)', image=mask_png,
                bounds=[[bottom, left], [top, right]], opacity=1.0, zindex=3).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(out_html)
        return out_html



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Desert visualization utilities")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--html", default="map_desert.html")
    ap.add_argument("--png", default="out/superval_overlay.png")
    args = ap.parse_args()

    OUT = args.outdir
    viz = DesertViz()

    D  = xr.open_dataarray(os.path.join(OUT, "desertness_mean.nc"))
    ST = xr.open_dataarray(os.path.join(OUT, "supertrue_mask.nc")).astype(bool)
    UD = xr.open_dataarray(os.path.join(OUT, "undetermined_mask.nc")).astype(bool)

   
    cls_path_final = os.path.join(OUT, "desert_class_final.nc")
    cls_path2 = os.path.join(OUT, "desert_class2.nc")
    CLS = None
    if os.path.exists(cls_path_final):
        CLS = xr.open_dataarray(cls_path_final)
    elif os.path.exists(cls_path2):
        CLS = xr.open_dataarray(cls_path2)

   
    try:
        SF = xr.open_dataarray(os.path.join(OUT, "superfalse_mask.nc")).astype(bool)
    except Exception:
        SF = (~ST) & (~UD)

    
    viz.save_superval_overlay_png(D, ST, UD, SF, args.png)
    print("Saved:", args.png)

    
    if _HAVE_FOLIUM and CLS is not None:
        out_html = viz.save_interactive_map(D, CLS, args.html)
        print("Saved:", out_html)
    else:
        if not _HAVE_FOLIUM:
            print("[note] folium 未安装，跳过交互地图。pip install folium")
        if CLS is None:
            print("[note] 未找到最终分类 (desert_class_final.nc / desert_class2.nc)，跳过交互地图。")
