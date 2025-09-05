from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Iterable
from datetime import datetime
import pathlib
import yaml

class ValidationError(Exception):
    """illegal configuration"""

@dataclass(frozen=True)
class ValidatedConfig:
    bbox: Tuple[float, float, float, float]          # (lat_min, lat_max, lon_min, lon_max)
    period: Tuple[str, str]                          # ("YYYY-MM", "YYYY-MM")
    output_dir: Optional[pathlib.Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)  

class ConfigValidator:

    def __init__(self, *, allowed_vars: Optional[Iterable[str]] = None):
        self.allowed_vars = set(allowed_vars) if allowed_vars else None

    def validate_yaml(self, path: str | pathlib.Path) -> ValidatedConfig:
        p = pathlib.Path(path)
        if not p.exists():
            raise ValidationError(f"config not found：{p.resolve()}")
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as e: 
            raise ValidationError(f"read YAML fail：{e}") from e
        vc = self.validate_dict(cfg)
        return vc

    def validate_dict(self, cfg: Dict[str, Any]) -> ValidatedConfig:
        if not isinstance(cfg, dict):
            raise ValidationError

        region = cfg.get("region") or {}
        bbox = region.get("bbox")
        period = region.get("period") or cfg.get("period")
        output_dir = (cfg.get("output", {}) or {}).get("dir")

        bbox_t = self._validate_bbox(bbox)
        period_t = self._validate_period(period)

        self._maybe_validate_fuzzy(cfg.get("fuzzy"))
        self._maybe_validate_supervaluation(cfg.get("supervaluation"))

        if "features" in cfg and self.allowed_vars:
            self._maybe_validate_vars(cfg["features"], self.allowed_vars)

        outdir_path = pathlib.Path(output_dir) if output_dir else None
        extras = {k: v for k, v in cfg.items() if k not in ("region", "output")}

        return ValidatedConfig(bbox=bbox_t, period=period_t, output_dir=outdir_path, extra=extras)

    @staticmethod
    def _validate_bbox(bbox: Any) -> Tuple[float, float, float, float]:
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValidationError("region.bbox need [lat_min, lat_max, lon_min, lon_max] ")
        try:
            lat_min, lat_max, lon_min, lon_max = (float(b) for b in bbox)
        except Exception:
            raise ValidationError("region.bbox have to be float")

        if not (-90.0 <= lat_min < lat_max <= 90.0):
            raise ValidationError(f"illigeal: [{lat_min}, {lat_max}]need -90 ≤ lat_min < lat_max ≤ 90。")
        if not (-180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0):
            raise ValidationError(f"range [-180, 180]：received [{lon_min}, {lon_max}]。")
        if lon_min == lon_max:
            raise ValidationError
        return (lat_min, lat_max, lon_min, lon_max)

    @staticmethod
    def _validate_period(period: Any) -> Tuple[str, str]:
        if not (isinstance(period, (list, tuple)) and len(period) == 2):
            raise ValidationError('region.period 需要形如 ["YYYY-MM", "YYYY-MM"]。')
        start, end = (str(period[0]).strip(), str(period[1]).strip())

        def parse_ym(s: str) -> datetime:
            try:
                return datetime.strptime(s, "%Y-%m")
            except ValueError as e:
                raise ValidationError(f"时间格式非法：{s}（必须为 YYYY-MM 且月份 01–12）") from e

        dt_start, dt_end = parse_ym(start), parse_ym(end)
        if dt_start > dt_end:
            raise ValidationError(f"period 起止顺序错误：{start} > {end}")
        return (start, end)

    @staticmethod
    def _maybe_validate_fuzzy(fz: Any) -> None:
        if not fz:
            return
        for key in ("phi_s", "ai_z"):
            if key in fz and isinstance(fz[key], dict):
                a, b = fz[key].get("a"), fz[key].get("b")
                if a is None or b is None:
                    raise ValidationError(f"fuzzy.{key}  need a and b")
                try:
                    a, b = float(a), float(b)
                except Exception:
                    raise ValidationError(f"fuzzy.{key}.a/b have to ne number")
                if not (a < b):
                    raise ValidationError(f"fuzzy.{key} have to satisfiy a < b")
        if "combine" in fz and not str(fz["combine"]).strip():
            raise ValidationError("fuzzy.combine cant be null")

    @staticmethod
    def _maybe_validate_supervaluation(sv: Any) -> None:
        if not sv:
            return
        th = sv.get("thresholds", {})
        st = th.get("supertrue", 0.6)
        sf = th.get("superfalse", 0.4)
        try:
            st, sf = float(st), float(sf)
        except Exception:
            raise ValidationError("supervaluation.thresholds need value")
        if not (0.0 <= sf < st <= 1.0):
            raise ValidationError(f"supervaluation need 0 ≤ superfalse < supertrue ≤ 1")

        prec = sv.get("precisifications", {})
        for name, items in prec.items():
            if not isinstance(items, (list, tuple)) or not items:
                raise ValidationError(f"supervaluation.precisifications.{name} cant be null")
            for i, ab in enumerate(items):
                if not isinstance(ab, dict) or "a" not in ab or "b" not in ab:
                    raise ValidationError(f"{name}[{i}] need {{a:..., b:...}}。")
                try:
                    a, b = float(ab["a"]), float(ab["b"])
                except Exception:
                    raise ValidationError(f"{name}[{i}]  a/b wrong format。")
                if not (a < b):
                    raise ValidationError(f"{name}[{i}] need a < b")

    @staticmethod
    def _maybe_validate_vars(features: Any, allowed: set[str]) -> None:
        if not features:
            return
        vars_requested = set()
        if isinstance(features, dict):
            for v in features.keys():
                if isinstance(v, str):
                    vars_requested.add(v)
        elif isinstance(features, (list, tuple)):
            vars_requested |= {str(v) for v in features}
        unknown = vars_requested - allowed
        if unknown:
            raise ValidationError(f"unkown：{sorted(unknown)}；allowed：{sorted(allowed)}")
