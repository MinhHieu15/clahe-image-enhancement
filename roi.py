"""Find objects in a grayscale image and return one ROI per object."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

from clahe import clahe


@dataclass
class ROI:
    label: int
    bbox: tuple          # (row0, col0, row1, col1)
    centroid: tuple      # (row, col)
    area: int
    diameter: float      # equivalent circle diameter (px)
    circularity: float   # 4*pi*area / perimeter^2, 1.0 for a circle
    mean_intensity: float


def segment(gray, use_clahe=True, clip=2.0, grid=8, blur=1.5, min_area=80, max_area=None,
            split_touching=True, min_distance=7, min_circularity=0.0, dark_objects=False):
    """Return (label image, list of ROI, preprocessed image)."""
    img = clahe(gray, clip, (grid, grid)) if use_clahe else gray
    smooth = filters.gaussian(img, sigma=blur, preserve_range=True)
    t = filters.threshold_otsu(smooth)
    mask = smooth < t if dark_objects else smooth > t
    mask = morphology.opening(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, max_size=min_area)

    if split_touching:
        dist = ndi.distance_transform_edt(mask)
        peaks = peak_local_max(dist, min_distance=min_distance, labels=measure.label(mask), exclude_border=False)
        markers = np.zeros(mask.shape, int)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        labels = segmentation.watershed(-dist, markers, mask=mask)
    else:
        labels = measure.label(mask)

    rois = []
    for r in measure.regionprops(labels, intensity_image=gray):
        perim = r.perimeter if r.perimeter > 0 else 1
        circ = 4 * np.pi * r.area / perim ** 2
        if r.area < min_area or (max_area and r.area > max_area) or circ < min_circularity:
            labels[labels == r.label] = 0
            continue
        rois.append(ROI(r.label, r.bbox, tuple(np.round(r.centroid, 1)), int(r.area),
                        round(float(r.equivalent_diameter_area), 2), round(float(circ), 3),
                        round(float(r.intensity_mean), 1)))
    return labels, rois, img
