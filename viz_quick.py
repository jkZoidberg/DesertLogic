
import matplotlib
matplotlib.use("Agg")  # 服务器/无界面环境
import matplotlib.pyplot as plt
import xarray as xr

def save_quick_png(da, out_png_path):
    ax = plt.figure(figsize=(6,4)).gca()
    im = da.plot(ax=ax, add_colorbar=True)
    ax.set_title(da.name or "layer")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160)
    plt.close()

from viz_quick import save_quick_png
D = xr.open_dataarray("out/desertness_mean.nc")
save_quick_png(D, "desertness_mean.png")
print("D min/max:", float(D.min()), float(D.max()))



ST = xr.open_dataarray("out/supertrue_mask.nc")
UD = xr.open_dataarray("out/undetermined_mask.nc")
print("supertrue%:", float(ST.mean()*100), "  undetermined%:", float(UD.mean()*100))