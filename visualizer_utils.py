import os
import numpy as np
from config import OPTIONS, pcds_dir, gts_dir, preds_dir


def quantize_xyz(pts):
    min_x, max_x = OPTIONS["quantize"]["range_x"]
    min_y, max_y = OPTIONS["quantize"]["range_y"]
    min_z, max_z = OPTIONS["quantize"]["range_z"]
    grid_sz = OPTIONS["quantize"]["grid_sz"]

    dx = (max_x - min_x) / grid_sz[0]
    dy = (max_y - min_y) / grid_sz[1]
    dz = (max_z - min_z) / grid_sz[2]

    return np.stack(
        [
            (pts[:, 0] - min_x) / dx,
            (pts[:, 1] - min_y) / dy,
            (pts[:, 2] - min_z) / dz,
        ],
        axis=1,
    )


def get_point_colors(gt, pred):
    point_colors = np.zeros((gt.size, 3), np.float32)
    COLORS = {
        "purple": (0.5, 0, 0.5),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "red": (1, 0, 0),
        "gray": (30 / 255, 30 / 255, 30 / 255),
    }

    if OPTIONS["confusion_vis"]["on"]:
        unlabeled = gt == 0
        tp = (gt == 251) & (pred == 251)
        fp = (gt != 251) & (pred == 251)
        fn = (gt == 251) & (pred != 251)
        tn = (gt == 9) & (pred == 9)

        point_colors[unlabeled] = COLORS["purple"]
        point_colors[tp] = COLORS["green"]
        point_colors[fp] = COLORS["blue"]
        point_colors[fn] = COLORS["red"]
        point_colors[tn] = COLORS["gray"]

    else:  # 라벨만 시각화
        unlabeled = gt == 0
        moving = pred == 251
        static = pred == 9

        point_colors[unlabeled] = COLORS["purple"]
        point_colors[moving] = COLORS["green"]
        point_colors[static] = COLORS["gray"]

    return point_colors


def load_one_scene(fid):
    pcd, gt, pred = None, None, None

    pcd_path = os.path.join(pcds_dir, f"{fid}.bin")
    pcd = np.fromfile(pcd_path, np.float32).reshape(-1, 4)[:, :3]

    gt_path = os.path.join(gts_dir, f"{fid}.label")
    gt = np.fromfile(gt_path, np.uint32) & 0xFFFF

    dist = np.linalg.norm(pcd[:, :2], axis=1)
    keep_dist = (dist >= OPTIONS["range_crop"]["min"]) & (dist <= OPTIONS["range_crop"]["max"])

    pcd = pcd[keep_dist]
    gt = gt[keep_dist]

    if OPTIONS["confusion_vis"]["on"]:
        pred_path = os.path.join(preds_dir, f"{fid}.label")
        pred = np.fromfile(pred_path, np.uint32) & 0xFFFF
        pred = pred[keep_dist]

    if OPTIONS["quantize"]["on"]:
        q_pcd = quantize_xyz(pcd)
        # 한 복셀에 있는 여러 점은 한 점으로 매핑
        vox_int = np.floor(q_pcd).astype(int)
        _, uniq = np.unique(vox_int, axis=0, return_index=True)
        pcd = q_pcd[uniq]
        gt = gt[uniq]

        if OPTIONS["confusion_vis"]["on"]:
            pred = pred[uniq]

    if OPTIONS["2d"]["on"]:
        pcd[:, 2] = 0

    point_colors = get_point_colors(gt, pred)
    return pcd, point_colors
