"""
Two applications of the ROI pipeline on real photographs (scikit-image sample data):

  coins   - count the 24 coins in a photo with uneven lighting, with and without CLAHE
  nuclei  - count cell nuclei in a fluorescence microscopy image and flag dividing (mitotic) cells

Everything is written to results/.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import color, data

from roi import segment

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COINS_TRUE = 24


def draw(ax, img, labels, rois, title, highlight=()):
    ax.imshow(color.label2rgb(labels, img, alpha=0.25, bg_label=0))
    for r in rois:
        r0, c0, r1, c1 = r.bbox
        col = "red" if r.label in highlight else "yellow"
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=col, lw=1))
    ax.set_title(title)
    ax.axis("off")


def save_rois(path, rois, extra=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["id", "row", "col", "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1", "area_px", "diameter_px",
                "circularity", "mean_intensity"]
        w.writerow(head + (list(extra) if extra else []))
        for n, r in enumerate(rois, 1):
            row = [n, *r.centroid, *r.bbox, r.area, r.diameter, r.circularity, r.mean_intensity]
            w.writerow(row + ([extra[k][r.label] for k in extra] if extra else []))


def coins():
    img = data.coins()
    lab_plain, plain, _ = segment(img, use_clahe=False, min_area=600, split_touching=False, min_circularity=0.6)
    lab, rois, enhanced = segment(img, use_clahe=True, grid=4, min_area=600, min_distance=26, min_circularity=0.6)
    rois.sort(key=lambda r: (round(r.centroid[0] / 60), r.centroid[1]))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input photo")
    axes[0].axis("off")
    draw(axes[1], img, lab_plain, plain, f"Otsu threshold only: {len(plain)} / {COINS_TRUE}")
    draw(axes[2], img, lab, rois, f"CLAHE + Otsu + watershed: {len(rois)} / {COINS_TRUE}")
    for n, r in enumerate(rois, 1):
        axes[2].text(r.centroid[1], r.centroid[0], str(n), color="white", fontsize=8, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6, lw=0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "coins_detection.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(4, 6, figsize=(10, 7))
    for ax, (n, r) in zip(axes.ravel(), enumerate(rois, 1)):
        r0, c0, r1, c1 = r.bbox
        ax.imshow(img[r0:r1, c0:c1], cmap="gray")
        ax.set_title(f"#{n}  d={r.diameter:.0f}px", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(rois):]:
        ax.axis("off")
    fig.suptitle("Extracted ROIs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "coins_roi_crops.png"), dpi=110)
    plt.close(fig)

    save_rois(os.path.join(OUT, "coins_rois.csv"), rois)
    return [("coins", "Otsu only", len(plain), COINS_TRUE), ("coins", "CLAHE + Otsu + watershed", len(rois), COINS_TRUE)]


def nuclei():
    img = data.human_mitosis()
    lab, rois, _ = segment(img, use_clahe=True, grid=8, blur=1.0, min_area=30, min_distance=5)
    inten = np.array([r.mean_intensity for r in rois])
    # dividing cells have condensed chromosomes: much brighter and smaller than the rest
    thr = np.median(inten) + 3 * 1.4826 * np.median(np.abs(inten - np.median(inten)))
    mitotic = {r.label for r in rois if r.mean_intensity > thr}

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Fluorescence microscopy input")
    axes[0].axis("off")
    draw(axes[1], img, lab, rois, f"{len(rois)} nuclei, {len(mitotic)} flagged as mitotic (red)", highlight=mitotic)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nuclei_detection.png"), dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist([r.area for r in rois], bins=40, color="steelblue")
    axes[0].set_xlabel("Nucleus area (px)")
    axes[0].set_ylabel("Count")
    axes[1].hist(inten, bins=40, color="gray")
    axes[1].axvline(thr, color="red", linestyle="--", label=f"mitosis threshold {thr:.0f}")
    axes[1].set_xlabel("Mean intensity")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nuclei_stats.png"), dpi=110)
    plt.close(fig)

    save_rois(os.path.join(OUT, "nuclei_rois.csv"), rois,
              extra={"mitotic": {r.label: int(r.label in mitotic) for r in rois}})
    areas = np.array([r.area for r in rois])
    print(f"nuclei: {len(rois)}, mitotic: {len(mitotic)} ({100 * len(mitotic) / len(rois):.1f} %), "
          f"median area {np.median(areas):.0f} px, threshold {thr:.1f}")
    return [("nuclei", "CLAHE + Otsu + watershed", len(rois), ""), ("nuclei (mitotic)", "intensity > median + 3 MAD", len(mitotic), "")]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = coins() + nuclei()
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "method", "detected", "true_count"])
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
