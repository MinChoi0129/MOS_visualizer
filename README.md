# MOS Visualizer

Open3D 기반 Moving Object Segmentation 포인트클라우드 시각화 도구.

## 설치

```bash
pip install git+https://github.com/MinChoi0129/MOS_visualizer.git
```

개발 모드:
```bash
git clone https://github.com/MinChoi0129/MOS_visualizer.git
cd MOS_visualizer
pip install -e .
```

## 사용법

```bash
mos-vis --config config.yaml
```

## 설정 (config.yaml)

```yaml
# 시각화 모드: pcd | gt | pred | confusion | feature
mode: pred

# 경로
pcds_dir: "/path/to/velodyne"
gts_dir: "/path/to/labels"
preds_dir: "/path/to/predictions"
imgs_dir: ""        # 카메라 이미지 (빈 문자열이면 비활성)
npy_file: ""        # feature 모드 전용

# 옵션
idx: 0              # 시작 프레임
json_path: ""       # Open3D 카메라 파라미터 json
range_crop: [2.0, 50.0]
flatten_2d: false
npy_norm: percentile  # minmax | sigmoid | percentile
```

### 모드 설명

| 모드 | 설명 | 필요 경로 |
|------|------|-----------|
| `pcd` | 포인트클라우드만 (흰색) | `pcds_dir` |
| `gt` | GT 라벨 색상 | `pcds_dir`, `gts_dir` |
| `pred` | 예측 라벨 색상 | `pcds_dir`, `preds_dir` |
| `confusion` | GT vs Pred confusion matrix | `pcds_dir`, `gts_dir`, `preds_dir` |
| `feature` | 딥러닝 피쳐맵 viridis 히트맵 | `npy_file` |

## 조작

- `N` : 다음 프레임
- `B` : 이전 프레임
- 마우스 : 3D 회전/줌
