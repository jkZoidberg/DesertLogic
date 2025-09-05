# save_quick_png.py
import numpy as np, rasterio, matplotlib.pyplot as plt

def save_quick_png(tif_path, out_png, cmap="terrain", title=None):
    with rasterio.open(tif_path) as src:
        a = src.read(1).astype("float32")
        if src.nodata is not None:
            a = np.where(a == src.nodata, np.nan, a)
        a = np.where(a == -500.0, np.nan, a)  # 海域遮掉
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

    plt.figure(figsize=(7,5))
    plt.imshow(a, extent=extent, origin="upper", cmap=cmap)
    plt.colorbar()
    if title: plt.title(title)
    plt.xlabel("Lon"); plt.ylabel("Lat")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print("saved:", out_png)

save_quick_png(
    r"F:\Leeds\Project\Program\DesertLogic\DEM\ACE2_TIF\heights\15N000E_3S.tif",
    r"F:\Leeds\Project\Program\DesertLogic\DEM\plot_height_15N000E.png",
    title="ACE2 Height 15N000E"
)