"""
So sánh: ảnh gốc | Histogram Equalization (HE) | CLAHE tự viết | CLAHE OpenCV
kèm histogram và chỉ số PSNR giữa CLAHE tự viết và OpenCV.

Chạy:
    python demo.py                    # dùng ảnh tương phản thấp tự sinh
    python demo.py --image anh.jpg    # dùng ảnh của bạn
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from clahe import clahe


def make_low_contrast_image(size=512):
    """Sinh ảnh thử: vùng tối/sáng không đều, chi tiết tương phản thấp."""
    y, x = np.mgrid[0:size, 0:size] / size
    base = 60 + 60 * x                          # chiếu sáng không đều
    pattern = 12 * np.sin(40 * x) * np.sin(40 * y)
    circles = 15 * (((x - 0.3) ** 2 + (y - 0.3) ** 2) < 0.02)
    noise = np.random.default_rng(0).normal(0, 3, (size, size))
    return np.clip(base + pattern + circles + noise, 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="đường dẫn ảnh đầu vào")
    ap.add_argument("--clip", type=float, default=2.0, help="clip limit")
    ap.add_argument("--grid", type=int, default=8, help="số tile mỗi chiều")
    ap.add_argument("--out", default="results/comparison.png")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"Không đọc được ảnh: {args.image}")
    else:
        img = make_low_contrast_image()

    he = cv2.equalizeHist(img)
    ours = clahe(img, args.clip, (args.grid, args.grid))
    ref = cv2.createCLAHE(clipLimit=args.clip, tileGridSize=(args.grid, args.grid)).apply(img)
    print(f"PSNR (CLAHE tự viết vs OpenCV): {psnr(ours, ref):.2f} dB")

    images = [("Ảnh gốc", img), ("Histogram Equalization", he),
              ("CLAHE (tự viết)", ours), ("CLAHE (OpenCV)", ref)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for k, (title, im) in enumerate(images):
        axes[0, k].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, k].set_title(title)
        axes[0, k].axis("off")
        axes[1, k].hist(im.ravel(), bins=256, range=(0, 255), color="steelblue")
        axes[1, k].set_xlim(0, 255)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("Đã lưu", args.out)
    plt.show()


if __name__ == "__main__":
    main()
