"""포인트클라우드 → Range View 투영"""

import numpy as np


def make_range_view(pcd, colors, H=128, W=1024):
    """포인트클라우드를 range view (H, W, 3) BGR 이미지로 투영"""
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    # KITTI HDL-64E: fov_up=3°, fov_down=-25°
    fov_up, fov_down = np.radians(3.0), np.radians(-25.0)
    fov = fov_up - fov_down

    azimuth = -np.arctan2(y, x)
    elevation = np.arcsin(z / np.clip(r, 1e-8, None))

    u = ((azimuth / np.pi + 1.0) * 0.5 * W).astype(np.int32) % W
    v = ((fov_up - elevation) / fov * H).astype(np.int32)
    v = np.clip(v, 0, H - 1)

    # 가까운 점이 위에 오도록: 먼 점 먼저, 가까운 점 나중에 덮어쓰기
    order = np.argsort(-r)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    bgr = (colors[order][:, ::-1] * 255).astype(np.uint8)
    img[v[order], u[order]] = bgr

    return img
