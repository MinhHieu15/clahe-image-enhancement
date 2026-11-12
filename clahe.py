"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) viết từ đầu bằng NumPy.

Các bước:
1. Chia ảnh thành lưới tile (vd 8x8).
2. Tính histogram mỗi tile, cắt (clip) các bin vượt ngưỡng clip_limit
   và phân phối lại phần dư đều cho mọi bin -> hạn chế khuếch đại nhiễu.
3. Tính hàm ánh xạ (CDF) cho từng tile.
4. Nội suy song tuyến tính giữa 4 tile lân cận để tránh hiệu ứng "ô vuông".
"""
import numpy as np


def _tile_mapping(tile, clip_limit, n_bins=256):
    hist = np.bincount(tile.ravel(), minlength=n_bins).astype(np.float64)
    # clip_limit tính theo bội số của giá trị trung bình mỗi bin (giống OpenCV)
    limit = max(clip_limit * tile.size / n_bins, 1.0)
    excess = np.sum(np.maximum(hist - limit, 0))
    hist = np.minimum(hist, limit) + excess / n_bins
    cdf = np.cumsum(hist)
    return np.clip(np.round((cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1) * 255), 0, 255)


def clahe(img, clip_limit=2.0, grid=(8, 8)):
    """Áp dụng CLAHE cho ảnh xám uint8."""
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("Cần ảnh xám kiểu uint8")
    h, w = img.shape
    gy, gx = grid
    th, tw = int(np.ceil(h / gy)), int(np.ceil(w / gx))

    # Đệm ảnh (phản chiếu) để chia hết cho kích thước tile
    padded = np.pad(img, ((0, th * gy - h), (0, tw * gx - w)), mode="reflect")

    maps = np.empty((gy, gx, 256))
    for i in range(gy):
        for j in range(gx):
            maps[i, j] = _tile_mapping(padded[i * th:(i + 1) * th, j * tw:(j + 1) * tw], clip_limit)

    # Toạ độ của mỗi pixel so với tâm các tile
    yc = (np.arange(h) + 0.5) / th - 0.5
    xc = (np.arange(w) + 0.5) / tw - 0.5
    y0 = np.clip(np.floor(yc).astype(int), 0, gy - 1)
    x0 = np.clip(np.floor(xc).astype(int), 0, gx - 1)
    y1 = np.clip(y0 + 1, 0, gy - 1)
    x1 = np.clip(x0 + 1, 0, gx - 1)
    wy = np.clip(yc - y0, 0, 1)[:, None]
    wx = np.clip(xc - x0, 0, 1)[None, :]

    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    v = img
    out = ((1 - wy) * (1 - wx) * maps[Y0, X0, v] + (1 - wy) * wx * maps[Y0, X1, v]
           + wy * (1 - wx) * maps[Y1, X0, v] + wy * wx * maps[Y1, X1, v])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
