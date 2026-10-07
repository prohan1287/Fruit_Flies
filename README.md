# Real-time Detailed Video Analysis of Fruit Flies — reproduction

UE24CS352A Machine Learning mini-project. A straight reproduction of Steven Herbst,
*Real-time Detailed Video Analysis of Fruit Flies* (CS229, Stanford, Fall 2018):
a CPU-only pipeline that finds the two flies in each 1530×1530 frame of a courtship video,
tells male from female, gets each fly's 0–360° heading, and measures the male's wing angles.

```
frame → FlyCount → ♂♀ vs ♀♂ → Orientation → WingAngle
        decision    standardize  HOG(3780) →    HOG(3564) →
        tree on     + logistic   PCA-15 →       PCA-40 →
        area        regression   logistic reg.  linear regression
```

No neural networks and no GPU. Every value the paper does not give is marked `# UNSPECIFIED:` in the code.
Those values come from the author's code. They are listed, along with the paper/poster discrepancies,
in [NOTES.md](NOTES.md).

## Results

### Table 1: model accuracy (`results/table1.md`)

| Model | Train error (ours / paper) | Test error (ours / paper) | N_train (ours / paper) | N_test (ours / paper) |
|---|---|---|---|---|
| FlyCount | 0.0% / 0.0% | 0.0% / 0.0% | 759 / 759 | 253 / 253 |
| ♂♀ vs ♀♂ | 0.0% / 0.2% | 0.7% / 0.7% | 423 / 423 | 141 / 141 |
| ♂ orientation | 0.0% / 0.0% | 0.0% / 0.0% | 358 / 358 | 120 / 120 |
| ♀ orientation | 0.8% / 0.8% | 0.0% / 0.0% | 261 / 259 | 87 / 87 |
| WingAngle | σ 2.35° / 2.06° | σ 2.18° / 2.92° | 354 / 354 | 118 / 118 |

- WingAngle mean test error: 0.011° (paper 0.592°).
- ♂♀ vs ♀♂ without the aspect-ratio features: 2.1% test error (poster 1.4%).
- FlyCount tree splits: 1112.5 and 10442.5 (paper Fig. 4: 1413.5 and 10442.5).

### Table 2: timing (`results/table2.md`)

Our machine: Intel Core i7-13700HX, 16 GB RAM, no GPU. Paper: 2.8 GHz Intel Core i7, 16 GB RAM, no GPU.
Median of 5 passes over `test4.mp4`: **72.5 FPS** (13.8 ms/frame, 2.4× the 30 FPS source rate) against the paper's 84 FPS. Per-stage times
were within about 2× of the paper's: I/O 4.3 ms, FlyCount 4.5 ms, ♂♀ 0.8 ms, Orientation 1.7 ms, WingAngle 2.5 ms.
On this laptop, separate passes over the same video range from about 22 to 72 FPS depending on background
load and power state. `profile.py` therefore reports the median of 5 passes and lists every pass.

### Figures (`results/`)

| File | Paper |
|---|---|
| `fig4_tree.png` | Fig. 4, FlyCount decision tree |
| `fig8_orientation_pca.png` | Fig. 8, female orientation HOG, PC1 vs PC2, normal vs flipped |
| `fig9_wing_pc1.png` | Fig. 9, wing angle vs PC1 |
| `fig10_wing_time.png` | Fig. 10, male wing angles over time on `test4.mp4`. Right/left correlation −0.69 (paper −0.64). |
| `poster_explained_variance.png` | Poster, cumulative explained variance. 15 PCs give 61% (♂) and 65% (♀) for orientation; paper ≈60%. |

## Setup

Python 3.14 on Windows 11 was used. Any recent Python 3 should work.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`opencv-python` must stay below 5, because OpenCV 5 removed `cv2.HOGDescriptor`.

### Data

The data is not in this repo (about 2 GB). Download the author's `input` folder from the link in
https://github.com/sgherbst/cs229-project (Dropbox:
https://www.dropbox.com/sh/78inyvw2ouut74a/AACc1DYrC1G0UxujwT-6ryRKa?dl=0) and place its two subfolders
in either of these locations:

```
ml-mini/images/   (326 LabelMe-annotated .bmp frames in 5 subfolders)
ml-mini/videos/   (test1.mp4 … test5.mp4)
```

or `ml-mini/input/images` and `ml-mini/input/videos`. Both locations are gitignored. Check the data with:

```powershell
python src/data.py      # should report 326 annotated frames
```

## Running

All commands run from the repo root.

| Step | Command | Output |
|---|---|---|
| Train all four models (about 40 s) | `python src/train.py` | `models/*.joblib`, `cache/` |
| Retrain without re-extracting features | `python src/train.py --cached` | |
| Table 1 | `python src/evaluate.py` | `results/table1.md` |
| Table 2 + FPS | `python src/profile.py -i test4.mp4 -n 5` | `results/table2.md` |
| Figures 4, 8, 9, 10 | `python src/figures.py -i test4.mp4` | `results/*.png` |
| Live demo | `python src/demo.py -i test4.mp4` | window; `space` pauses, `q` quits |
| Save the demo as video | `python src/demo.py -i test4.mp4 --no-display --write-video` | `results/demo_test4.mp4` |

On Windows, set `PYTHONIOENCODING=utf-8` if the console can't print ♂/♀.
Trained models are committed, so `evaluate`, `profile`, `figures` and `demo` work without retraining.
Training is deterministic: seed 0 is used for the split, the tree and PCA.

In the demo overlay, red marks the male and blue the female. Each fly has an outline, a centre dot and a
heading arrow, and cyan arrows show the male's right and left wings. Magenta means the flies are touching
and form one contour.

## Code layout

```
src/
  data.py         LabelMe loading, contour ground truth, 25% held-out split (seed 0)
  imgproc.py      thresholds, contours, moment angle/axes, rotate/crop helpers
  flycount.py     Stage 1: contour area → Gini decision tree (0/1/2 flies)
  sex.py          Stage 2: area + aspect ratio of both contours → StandardScaler → logistic regression
  orientation.py  Stage 3: upright 128×64 crop → HOG 3780 → PCA-15 → logistic regression (♂ and ♀ models)
  wingangle.py    Stage 4: blur/threshold/erode → upright half-crop 96×80 → HOG 3564 → PCA-40 →
                  closed-form linear regression θ = (XᵀX)⁻¹Xᵀy
  pipeline.py     per-frame chaining of the four stages, with a per-stage profiler
  train.py        builds every stage's dataset (with augmentation), splits, fits, saves
  evaluate.py     Table 1, poster ablation, Fig. 4 tree text
  profile.py      Table 2 and throughput
  figures.py      Figs. 4, 8, 9, 10 and the explained-variance curves
  demo.py         live overlay on a test video
```

## How this differs from the paper

These are covered in detail in [NOTES.md](NOTES.md).
- **Test split.** The paper says "one third" is held out, but every Table 1 row is a 25% split. We use 25%,
  split per stage after augmentation, as the author's code does. As a result the augmented twin of a training
  example can end up in the test set.
- **Stage 4 thresholding.** The paper calls it "adaptive"; the author's code thresholds at the mean of the well
  minus 5, and we follow the code.
- **Throughput ratio.** The paper says 2.4× real time while the poster's 30 FPS source gives 2.8×. The test
  videos report 35 FPS, and 84/35 = 2.4.

## Credits

Method, data and reference implementation: Steven Herbst, https://github.com/sgherbst/cs229-project (MIT license).
