from dataclasses import dataclass
from typing import Tuple
import yaml


@dataclass
class Config:
    # 시각화 모드: "pcd" | "gt" | "pred" | "confusion" | "feature"
    mode: str = "pred"

    # 경로
    pcds_dir: str = ""
    gts_dir: str = ""
    preds_dir: str = ""
    imgs_dir: str = ""      # 빈 문자열이면 카메라 이미지 비활성
    npy_file: str = ""      # feature 모드 전용

    # 공통 옵션
    idx: int = 0
    json_path: str = ""
    range_crop: Tuple[float, float] = (2.0, 50.0)
    flatten_2d: bool = False

    # feature 모드 정규화: "minmax" | "sigmoid" | "percentile"
    npy_norm: str = "percentile"


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "range_crop" in data and isinstance(data["range_crop"], list):
        data["range_crop"] = tuple(data["range_crop"])
    return Config(**data)
