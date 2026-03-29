"""포인트클라우드/라벨 로드 + 색상 적용"""

import os
import numpy as np
from .coloring import color_pcd, color_gt, color_pred, color_confusion, color_feature


def _crop_range(range_crop, pcd, *arrays):
    dist = np.linalg.norm(pcd[:, :2], axis=1)
    rmin, rmax = range_crop
    keep = (dist >= rmin) & (dist <= rmax)
    return (pcd[keep], *(a[keep] for a in arrays))


def _load_bin(pcds_dir, fid):
    return np.fromfile(os.path.join(pcds_dir, f"{fid}.bin"), np.float32).reshape(-1, 4)[:, :3]


def _load_label(directory, fid, n_fallback):
    path = os.path.join(directory, f"{fid}.label")
    try:
        return np.fromfile(path, np.uint32) & 0xFFFF
    except FileNotFoundError:
        return np.zeros(n_fallback, dtype=np.uint32)


def load_scene(cfg, fid):
    pcd = _load_bin(cfg.pcds_dir, fid)
    n = pcd.shape[0]

    gt = _load_label(cfg.gts_dir, fid, n) if cfg.mode in ("gt", "confusion") else None
    pred = _load_label(cfg.preds_dir, fid, n) if cfg.mode in ("pred", "confusion") else None

    arrays = [a for a in (gt, pred) if a is not None]
    pcd, *arrays = _crop_range(cfg.range_crop, pcd, *arrays)
    it = iter(arrays)
    if gt is not None:
        gt = next(it)
    if pred is not None:
        pred = next(it)

    if cfg.flatten_2d:
        pcd[:, 2] = 0

    if cfg.mode == "pcd":
        colors = color_pcd(len(pcd))
    elif cfg.mode == "gt":
        colors = color_gt(gt)
    elif cfg.mode == "pred":
        colors = color_pred(pred)
    elif cfg.mode == "confusion":
        colors = color_confusion(gt, pred)

    return pcd, colors


def load_feature(cfg):
    data = np.load(cfg.npy_file)
    pcd = data[:, :3]
    feat = data[:, 3] if data.shape[1] >= 4 else np.linalg.norm(pcd[:, :2], axis=1)

    pcd, feat = _crop_range(cfg.range_crop, pcd, feat)
    if cfg.flatten_2d:
        pcd[:, 2] = 0

    return pcd, color_feature(feat, cfg.npy_norm)
