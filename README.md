# CLAHE Image Enhancement

CLAHE (Contrast Limited Adaptive Histogram Equalization) written from scratch in NumPy, compared with global histogram equalization and OpenCV's `cv2.createCLAHE`.

![Comparison](results/comparison.png)

Global histogram equalization uses the histogram of the whole image, so under uneven lighting it washes out bright areas and boosts noise in flat ones. CLAHE works per tile instead:

1. Split the image into tiles (8×8 by default) and equalize each one separately.
2. Clip each histogram bin at a limit and spread the excess over all bins, which limits how much contrast and noise a flat region can gain.
3. Blend the mappings of the four nearest tiles with bilinear interpolation so the tile edges don't show.

## Files
```
clahe.py   # clahe(img, clip_limit, grid)
demo.py    # original / HE / this CLAHE / OpenCV CLAHE, histograms and PSNR
results/   # output images
```

## Running
```bash
pip install -r requirements.txt
python demo.py
python demo.py --image photo.jpg --clip 3 --grid 8
```
Without `--image` it uses a synthetic low-contrast image. On that image the output matches OpenCV at 43 dB PSNR.
