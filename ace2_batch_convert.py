# ace2_batch_convert.py  —— 只保留最必要部分
# 用法示例（文件末尾也有示例）：
#   python ace2_batch_convert.py --in-root "F:/.../dedc-ace-v2_15N000E_3sec" --out-root "F:/.../ACE2_TIF" --overwrite

# --- 在导入 rasterio/pyproj 前：为 VSCode/绝对路径运行兜底注入 GDAL/PROJ ---
import os, sys
from pathlib import Path
def _ensure_gdal_proj_env():
    prefix = Path(sys.executable).parent.parent  # .../envs/torchGPU
    gdal = (prefix / "Library" / "share" / "gdal").resolve()
    proj = (prefix / "Library" / "share" / "proj").resolve()
    if gdal.exists(): os.environ.setdefault("GDAL_DATA", str(gdal))
    if proj.exists():
        os.environ.setdefault("PROJ_LIB",  str(proj))
        os.environ.setdefault("PROJ_DATA", str(proj))
_ensure_gdal_proj_env()

import argparse, gzip, re
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS as _CRS

# ---- 常量 ----
ROWS, COLS = 18000, 18000                 # 每幅 18000x18000（3″）
PIX_DEG = 15.0 / ROWS                     # 0.000833333333...
ACE2_VOID = -32768.0                      # 空洞
ACE2_SEA_MASK = -500.0                    # 海域掩膜（是否当 nodata 由参数决定）
_TILE_RE = re.compile(r"(?P<lat>\d{2})(?P<ns>[NS])(?P<lon>\d{3})(?P<ew>[EW])", re.IGNORECASE)

def _crs_wgs84():
    """优先 EPSG，失败则返回内嵌 WKT（不依赖 proj.db）。"""
    try:
        return _CRS.from_epsg(4326)
    except Exception:
        return _CRS.from_wkt(
            'GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
            'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
            'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
        )

def _parse_sw_from_token(token: str):
    m = _TILE_RE.fullmatch(token)
    if not m: return None
    lat = int(m.group("lat"));  lon = int(m.group("lon"))
    if m.group("ns").upper() == "S": lat = -lat
    if m.group("ew").upper() == "W": lon = -lon
    return float(lat), float(lon)  # (sw_lat, sw_lon)

def _find_tile_token(p: Path):
    # 文件名优先，找不到再看上级目录
    for part in [p.name] + [q.name for q in p.parents]:
        m = _TILE_RE.search(part.upper())
        if m: return m.group(0)
    return None

def _detect_type(p: Path):
    name = p.name.upper()
    for k in ("HEIGHT","SOURCE","QUALITY","CONFIDENCE"):
        if k in name: return k
    if "ACCURACY" in name: return "QUALITY"
    parent = p.parent.name.upper()
    if "HEIGHT" in parent: return "HEIGHT"
    if "SOURCE" in parent: return "SOURCE"
    if "QUALITY" in parent or "ACCURACY" in parent: return "QUALITY"
    if "CONFIDENCE" in parent: return "CONFIDENCE"
    return None

def _read_ace2_array(bin_path: Path, dtype_name: str) -> np.ndarray:
    """读取 .ACE2 或 .ACE2.gz 为 (ROWS, COLS) 数组。简洁实现：.gz 走内存解压（如内存不足再换 memmap 版）。"""
    if dtype_name == "HEIGHT":
        dtype = np.dtype("<f4")
    else:
        dtype = np.dtype("<i2")

    if bin_path.suffix.lower() == ".gz":
        with gzip.open(bin_path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=dtype, count=ROWS*COLS)
    else:
        arr = np.fromfile(bin_path, dtype=dtype, count=ROWS*COLS)

    if arr.size != ROWS*COLS:
        raise ValueError(f"Size mismatch: {bin_path}, got {arr.size}")
    return arr.reshape((ROWS, COLS))

def _out_path(in_path: Path, out_root: Path, dtype_name: str) -> Path:
    subdir = {"HEIGHT":"heights","SOURCE":"source","QUALITY":"quality","CONFIDENCE":"confidence"}[dtype_name]
    out_dir = out_root / subdir; out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.name
    U = stem.upper()
    if U.endswith(".ACE2.GZ"): stem = stem[:-len(".ACE2.GZ")]
    elif U.endswith(".ACE2"):  stem = stem[:-len(".ACE2")]
    return out_dir / f"{stem}.tif"

def _write_geotiff(arr: np.ndarray, sw_lat: float, sw_lon: float, out_path: Path, dtype_name: str, sea_as_nodata=False):
    west, north = sw_lon, sw_lat + 15.0
    profile = dict(
        driver="GTiff", height=ROWS, width=COLS, count=1,
        crs=_crs_wgs84(), transform=from_origin(west, north, PIX_DEG, PIX_DEG),
        tiled=True, blockxsize=512, blockysize=512, compress="LZW", bigtiff="YES",
    )
    if dtype_name == "HEIGHT":
        profile["dtype"] = "float32"; profile["nodata"] = ACE2_VOID
        data = arr.astype(np.float32, copy=True)
        if sea_as_nodata:
            data[data == ACE2_SEA_MASK] = np.nan
    else:
        profile["dtype"] = "int16"
        data = arr.astype(np.int16, copy=False)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)

def main():
    ap = argparse.ArgumentParser(description="Convert ACE2 .ACE2/.ACE2.gz to GeoTIFF (minimal)")
    ap.add_argument("--in-root",  required=True, help="ACE2 根目录（递归扫描）")
    ap.add_argument("--out-root", required=True, help="输出根目录")
    ap.add_argument("--datasets", nargs="*", default=["HEIGHT","SOURCE","QUALITY","CONFIDENCE"],
                    choices=["HEIGHT","SOURCE","QUALITY","CONFIDENCE"], help="要处理的类型")
    ap.add_argument("--overwrite", action="store_true", help="存在则覆盖")
    ap.add_argument("--sea-as-nodata", action="store_true", help="把海域 -500 也视为 nodata")
    args = ap.parse_args()

    in_root  = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    wanted = set(args.datasets)

    # 收集目标
    files = []
    for ext in ("*.ACE2","*.ACE2.GZ","*.ace2","*.ace2.gz"):
        for p in in_root.rglob(ext):
            t = _detect_type(p)
            if not t or t not in wanted: continue
            token = _find_tile_token(p)
            if not token: 
                print(f"[skip] 无法解析瓦片名: {p}"); 
                continue
            files.append((p, t, token))

    if not files:
        print("未找到可处理文件"); return
    print(f"[ACE2] 发现 {len(files)} 个文件，开始转换…")

    # 转换
    done = 0
    for src, dtype_name, token in files:
        sw = _parse_sw_from_token(token)
        if not sw:
            print(f"[skip] 瓦片名无效: {src}"); continue
        out_tif = _out_path(src, out_root, dtype_name)
        if out_tif.exists() and not args.overwrite:
            done += 1; continue
        try:
            arr = _read_ace2_array(src, dtype_name)
            _write_geotiff(arr, sw_lat=sw[0], sw_lon=sw[1], out_path=out_tif,
                           dtype_name=dtype_name, sea_as_nodata=args.sea_as_nodata)
            done += 1
        except Exception as e:
            print(f"[error] {src} -> {e}")

    print(f"[ACE2] 完成（成功 {done}/{len(files)}）。")

if __name__ == "__main__":
    # 你的实际路径（可直接改这里后运行：python ace2_batch_convert.py）
    # 也可以从命令行传参。
    if len(sys.argv) == 1:
        sys.argv = [
            "ace2_batch_convert.py",
            "--in-root",  r"F:\Leeds\Project\Program\DesertLogic\DEM\dedc-ace-v2_15N000E_3sec",
            "--out-root", r"F:\Leeds\Project\Program\DesertLogic\DEM\ACE2_TIF",
            "--overwrite"  # 如不想覆盖已有文件，删除这一行
        ]
    main()
