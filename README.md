# BIOM-Zad1 — Iris Segmentation with the Hough Circle Transform

Assignment for the Biometrics (BIOM) course: detecting the pupil, iris, and eyelids
in an eye image using the Hough Circle Transform, with an interactive OpenCV GUI
for tuning detection parameters and a simple evaluation pipeline based on
Intersection-over-Union (IoU).

## What it does

- **Preprocessing** (`preprocess_image`): grayscale conversion, optional CLAHE
  contrast enhancement, Gaussian blur, and Canny edge detection.
- **Circle detection** (`detect_circles` / `detect_pupil` / `detect_iris` /
  `detect_lids`): applies `cv2.HoughCircles` with parameters tuned separately for
  each structure, then filters the raw candidates:
  - **Pupil** — smallest circle near the image center.
  - **Iris** — circle larger than the pupil and centered close to it.
  - **Eyelids** (top/bottom) — large circles positioned above/below the iris.
- **Visualization**: detected circles are drawn on the image, color-coded by
  structure (pupil, iris, top lid, bottom lid).
- **Interactive GUI** (`setup_trackbars` / `update`): trackbars for padding,
  Gaussian blur kernel size, and all `HoughCircles` parameters (`dp`, `minDist`,
  `param1`, `param2`, `minRadius`, `maxRadius`), plus toggles for CLAHE and the
  Hough transform itself, so parameters can be tuned live.
- **Evaluation** (`compute_iou`, `evaluate_detection`): computes IoU between
  detected and annotated (ground-truth) circles and derives Precision, Recall,
  and F1-score using a configurable IoU threshold.
- **Grid search** (`grid_search`): brute-force search over a parameter grid for
  `HoughCircles`, scoring each combination by F1 against annotated data.

## Requirements

- Python 3
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- NumPy
- pandas
- Matplotlib

Install with:

```bash
pip install opencv-python numpy pandas matplotlib
```

## Data

The script expects:

- An eye image, referenced via `image_path` in [main.py](main.py) (default:
  `duhovky/013/L/S1013L01.jpg`).
- An annotation CSV file, `iris_annotation.csv`, with one row per image and 12
  numeric columns: `pupil (x, y, r)`, `iris (x, y, r)`, `bottom_lid (x, y, r)`,
  `top_lid (x, y, r)`.

Neither the dataset nor the annotation file is included in this repository —
place them alongside `main.py` (or update the paths) before running.

## Usage

```bash
python main.py
```

This opens three OpenCV windows (`Settings`, `Preprocessed`, `Blurred`, `Result`)
showing the live preprocessing/detection pipeline driven by the trackbars in
the `Settings` window.

**Keyboard shortcuts** (with a window focused):

- `c` — run full four-circle detection (pupil, iris, top lid, bottom lid) on
  the configured image, print the detected vs. annotated circles, and report
  Precision/Recall/F1-score.
- `Esc` — close all windows and exit.

## Notes

- The grid-search block at the bottom of [main.py](main.py) is commented out by
  default; uncomment it (and supply a dataset/ground truth) to sweep the
  `param_grid` and export results to `grid_search_results.csv`.
- Detection thresholds (tolerances, radius ranges) were tuned for a specific
  dataset/image resolution and may need adjustment for other images.
