"""포인트클라우드 색상 매핑"""

import numpy as np
import matplotlib.cm as cm

VIRIDIS_BG = np.array(cm.viridis(0.0)[:3])

LABEL_STATIC = 9
LABEL_MOVING = 251
LABEL_UNLABELED = 0

GT_STATIC_LABELS = [
    9, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
    40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99,
]
GT_MOVING_LABELS = [251, 252, 253, 254, 255, 256, 257, 258, 259]

COLORS = {
    "tp": (0, 1, 0),
    "fp": (0, 0, 1),
    "fn": (1, 0, 0),
    "tn": (30 / 255,) * 3,
    "moving": (0, 1, 0),
    "static": (30 / 255,) * 3,
    "unlabeled": (0.5, 0, 0.5),
    "white": (1, 1, 1),
}


def remap_labels(gt):
    for label in GT_STATIC_LABELS:
        gt[gt == label] = LABEL_STATIC
    for label in GT_MOVING_LABELS:
        gt[gt == label] = LABEL_MOVING
    return gt


def color_pcd(n):
    colors = np.zeros((n, 3), np.float32)
    colors[:] = COLORS["white"]
    return colors


def color_gt(gt):
    gt = remap_labels(gt)
    colors = np.zeros((len(gt), 3), np.float32)
    colors[gt == LABEL_MOVING] = COLORS["moving"]
    colors[gt == LABEL_STATIC] = COLORS["static"]
    colors[gt == LABEL_UNLABELED] = COLORS["unlabeled"]
    return colors


def color_pred(pred):
    colors = np.zeros((len(pred), 3), np.float32)
    colors[pred == LABEL_MOVING] = COLORS["moving"]
    colors[pred == LABEL_STATIC] = COLORS["static"]
    colors[pred == LABEL_UNLABELED] = COLORS["unlabeled"]
    return colors


def color_confusion(gt, pred):
    gt = remap_labels(gt)
    colors = np.zeros((len(gt), 3), np.float32)
    colors[(gt >= LABEL_MOVING) & (pred == LABEL_MOVING)] = COLORS["tp"]
    colors[(gt == LABEL_STATIC) & (pred == LABEL_MOVING)] = COLORS["fp"]
    colors[(gt >= LABEL_MOVING) & (pred == LABEL_STATIC)] = COLORS["fn"]
    colors[(gt == LABEL_STATIC) & (pred == LABEL_STATIC)] = COLORS["tn"]
    colors[gt == LABEL_UNLABELED] = COLORS["unlabeled"]
    return colors


def color_feature(feat, npy_norm="percentile"):
    if npy_norm == "sigmoid":
        norm = 1.0 / (1.0 + np.exp(-feat))
    elif npy_norm == "percentile":
        p1, p99 = np.percentile(feat, [1, 99])
        clipped = np.clip(feat, p1, p99)
        norm = (clipped - p1) / (p99 - p1) if p99 - p1 > 1e-8 else np.zeros_like(feat)
    else:  # minmax
        vmin, vmax = feat.min(), feat.max()
        norm = (feat - vmin) / (vmax - vmin) if vmax - vmin > 1e-8 else np.zeros_like(feat)
    return cm.viridis(norm)[:, :3].astype(np.float32)
