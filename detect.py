"""
Detect objects in your own image and export one ROI per object.

    python detect.py photo.jpg
    python detect.py photo.jpg --dark --min-area 200 --no-split
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import color, io

from examples import save_rois
from roi import segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="results/detect")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--min-circularity", type=float, default=0.0)
    ap.add_argument("--min-distance", type=int, default=7, help="min distance between object centres when splitting")
    ap.add_argument("--no-split", action="store_true", help="do not split touching objects")
    ap.add_argument("--dark", action="store_true", help="objects are darker than the background")
    args = ap.parse_args()

    img = io.imread(args.image)
    if img.ndim == 3:
        img = (color.rgb2gray(img[..., :3]) * 255).astype("uint8")
    labels, rois, _ = segment(img, use_clahe=not args.no_clahe, clip=args.clip, grid=args.grid,
                              min_area=args.min_area, split_touching=not args.no_split,
                              min_distance=args.min_distance, min_circularity=args.min_circularity,
                              dark_objects=args.dark)

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    fig, ax = plt.subplots(figsize=(10, 10 * img.shape[0] / img.shape[1]))
    ax.imshow(color.label2rgb(labels, img, alpha=0.25, bg_label=0))
    for n, r in enumerate(rois, 1):
        r0, c0, r1, c1 = r.bbox
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="yellow", lw=1))
        ax.text(c0, r0 - 2, str(n), color="yellow", fontsize=7)
    ax.set_title(f"{len(rois)} objects")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{name}_rois.png"), dpi=120)
    save_rois(os.path.join(args.out, f"{name}_rois.csv"), rois)
    print(f"{len(rois)} objects -> {args.out}")


if __name__ == "__main__":
    main()
