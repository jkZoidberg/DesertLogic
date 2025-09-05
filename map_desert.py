import sys, os
import numpy as np
import xarray as xr
import folium
from PIL import Image
from matplotlib import cm
from matplotlib import colors as mpl_colors

def to_png(arr, vmin=None, vmax=None, cmap_name='viridis', alpha=0.7):
    vmin = np.nanmin(arr) if vmin is None else vmin
    vmax = np.nanmax(arr) if vmax is None else vmax
    rgba = cm.get_cmap(cmap_name)(mpl_colors.Normalize(vmin=vmin, vmax=vmax)(arr))
    rgba[..., -1] = np.where(np.isnan(arr), 0, alpha)
    return Image.fromarray((rgba*255).astype(np.uint8))

def mask_to_png(mask, color=(255,0,0), alpha=0.45):
    h, w = mask.shape
    img = np.zeros((h,w,4), dtype=np.uint8)
    img[mask==1] = [*color, int(alpha*255)]
    return Image.fromarray(img, 'RGBA')

def bounds_from_coords(da):
    lat = da['lat'].values
    lon = da['lon'].values
    top, bottom = float(lat.max()), float(lat.min())
    left, right = float(lon.min()), float(lon.max())
    flipud = lat[0] < lat[-1]  

def main(d_path='out/desertness_mean.nc', cls_path='out/desert_class.nc'):
    D = xr.open_dataarray(d_path)
    left, bottom, right, top, flipud = bounds_from_coords(D)

    arr = D.values
    if flipud: arr = np.flipud(arr)
    heat = to_png(arr, vmin=0.0, vmax=1.0, alpha=0.7)
    heat_path = 'desertness_overlay.png'
    heat.save(heat_path)

    mask_path = None
    if os.path.exists(cls_path):
        CLS = xr.open_dataarray(cls_path)
        m = (CLS.values == 1).astype(np.uint8)
        if flipud: m = np.flipud(m)
        mask_img = mask_to_png(m, color=(255,0,0), alpha=0.45)
        mask_path = 'desert_mask_overlay.png'
        mask_img.save(mask_path)

    center = [(top+bottom)/2, (left+right)/2]
    m = folium.Map(location=center, zoom_start=5, tiles='CartoDB positron')
    folium.raster_layers.ImageOverlay(
        name='Desertness (0-1)',
        image=heat_path,
        bounds=[[bottom, left], [top, right]],
        opacity=1.0,
        zindex=2,
    ).add_to(m)
    if mask_path:
        folium.raster_layers.ImageOverlay(
            name='Desert mask (class=1)',
            image=mask_path,
            bounds=[[bottom, left], [top, right]],
            opacity=1.0,
            zindex=3,
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out_html = 'map_desert.html'
    m.save(out_html)
    print('Saved', out_html)

if len(sys.argv) >= 2:
    d = sys.argv[1]
    c = sys.argv[2] if len(sys.argv) >= 3 else 'out/desert_class.nc'
    main(d, c)
else:
    main()