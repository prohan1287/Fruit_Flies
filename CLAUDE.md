# CLAUDE.md — UE24CS352A ML Mini-Project

## Project

Reproduce **"Real-time Detailed Video Analysis of Fruit Flies"** (Steven Herbst, CS229 Fall 2018 Final Project, Stanford).

- Course: UE24CS352A — Machine Learning
- Team: Rohan Komperla (PES1UG24CS380), Dhanush Gowda AS (PES1UG24CS812)
- Sources of truth: the paper (`paper.pdf`) and the poster (`poster.pdf`). The guidelines document (`guidelines.pdf`) governs deliverables.

## Ground rules

1. **Straight reproduction only.** Implement exactly what the paper/poster describe. Do not add models, features, tuning, or improvements that are not in the sources.
2. **Do not fill gaps silently.** Where the paper/poster do not specify something (a hyperparameter, threshold value, preprocessing constant), do not invent a value. Mark it in code with `# UNSPECIFIED:` and list it in `NOTES.md`, then ask the team before choosing.
3. **Record discrepancies.** Where the paper and poster disagree, note both values in `NOTES.md` (see "Known discrepancies" below) and ask which to follow.
4. **Allowed libraries:** the ones the paper names — `scikit-learn`, `opencv-python`, `numpy`, `matplotlib`, `imutils`, `joblib`, `tqdm`. PyTorch/Keras are allowed by the team's earlier decision but are not needed: the paper uses no neural networks.
5. **No GPU.** The paper's throughput claim is CPU-only; keep it that way.

## Data

- Author's repo: https://github.com/sgherbst/cs229-project (MIT license).
- Data: the repo README links a ~2 GB Dropbox "input" folder with two subfolders, `images` (326 LabelMe-annotated frames) and `video` (15-min grayscale courtship video, cropped to 1530×1530; test clips `test1.mp4`–`test5.mp4`).
- Do not commit the data to our repo. Add `input/` to `.gitignore` and document the download step in the README.
- Labels (LabelMe JSON): head, abdomen, and a body point for each fly; male flies have extra wing points (labels seen in Fig. 2: `fh`, `fp`, `fa`, `mh`, `mp`, `mp2`, `ma`, `mw`).
- Split: hold out **one third** of the data as a test set, never used in training (Section 6).

## Pipeline to reproduce (4 ML stages)

Frame read (I/O) → FlyCount → ♂♀ vs ♀♂ → Orientation → WingAngle

### Stage 1 — FlyCount (poster: "Fly/NotFly")
- Threshold the image, keeping only the central fly body (removes background and appendages).
- Extract contours.
- Feature: contour area (single feature).
- Model: decision tree, Gini impurity criterion.
- Output: 0, 1, or 2 flies per contour (3 classes).
- Reference tree (Fig. 4): split at area ≤ 1413.5 → neither; then area ≤ 10442.5 → one, else both.

### Stage 2 — ♂♀ vs ♀♂ (poster: "MF vs. FM")
- Runs when there are two one-fly contours.
- Features (4): area and aspect ratio of both contours; aspect ratio from image moments.
- Preprocessing: standardize to zero mean, unit variance (`StandardScaler`).
- Model: logistic regression; output is whether the pair is ordered male-female or female-male.
- Augmentation: swap the order of the two contours and flip the label.
- Poster also reports an ablation: 1.4% test error without aspect-ratio features.

### Stage 3 — Orientation
- Mask the image to one fly's body.
- Angle from central image moments: θ ≈ ½·atan(2μ11 / (μ20 − μ02)). This has a 0/180° ambiguity.
- Rotate the fly upright by θ, crop/resample to **128×64** (poster: "64x128"), compute HOG (8×8 cells, 9 orientation bins) → **3780×1** descriptor.
- PCA → **15 components** (~60% variance explained).
- Logistic regression decides whether to add 180°.
- Trained **separately for male and female** flies.
- Augmentation: rotate images 180° and invert labels.

### Stage 4 — WingAngle (male only)
- ROI with the male fly and wings, oriented upright using Stage 3.
- Preprocessing: median blur, adaptive threshold, erosion (keeps wings, removes legs).
- Split into right and left halves; one wing per half.
- Crop/resample to **96×80** (poster: "80x96"), compute HOG → **3564×1** descriptor.
- PCA → **40 components**.
- Linear regression (closed form θ = (XᵀX)⁻¹Xᵀy) → wing angle, 0–90° relative to the body's major axis.
- Repeat for the left wing.

## Target results (Table 1 / Table 2)

| Model | Train error | Test error | N_train | N_test |
|---|---|---|---|---|
| FlyCount | 0.0% | 0.0% | 759 | 253 |
| ♂♀ vs ♀♂ | 0.2% | 0.7% | 423 | 141 |
| ♂ orientation | 0.0% | 0.0% | 358 | 120 |
| ♀ orientation | 0.8% | 0.0% | 259 | 87 |
| WingAngle | σ = 2.06° | σ = 2.92° (mean 0.592°) | 354 | 118 |

| Step | Runtime | % total |
|---|---|---|
| I/O | 3.9 ms | 32.5% |
| FlyCount | 2.2 ms | 18.3% |
| ♂♀ vs ♀♂ | 0.7 ms | 5.8% |
| Orientation | 1.9 ms | 15.8% |
| WingAngle | 3.3 ms | 27.5% |

- Throughput: **84.0 FPS** (11.9 ms/frame) on 1530×1530 frames, 2.8 GHz Intel Core i7, 16 GB RAM, no GPU. Source video is 30 FPS.
- Our hardware differs; report our measured FPS and per-stage timings alongside the paper's, and state the hardware.

Figures worth reproducing: Fig. 8 (PC1 vs PC2 of orientation HOG, normal vs flipped), Fig. 9 (wing angle vs PC1), Fig. 10 (wing angle over time; paper reports right/left cross-correlation of −0.64), and the explained-variance curves from the poster.

## Known discrepancies / gaps (log in NOTES.md, ask before resolving)

- Throughput ratio: paper says 84 FPS is "2.4x faster" than source video; poster says source is 30 FPS, which gives 2.8×.
- Image sizes written as H×W in the paper and W×H in the poster (128×64 vs 64×128; 96×80 vs 80×96).
- Stage names differ: "FlyCount" (paper) vs "Fly/NotFly" (poster).
- Unspecified: threshold values for Stage 1 body segmentation; median blur kernel, adaptive threshold block size/constant, erosion kernel/iterations for Stage 4; HOG block/stride parameters beyond 8×8 cells and 9 bins; logistic regression regularization/solver; decision tree depth limit; random seed and exact split procedure; how wing angle labels are computed from the annotated points.

## Repository layout (required by guidelines)

```
.
├── CLAUDE.md
├── README.md          # setup + how to run (required)
├── NOTES.md           # unspecified details, discrepancies, decisions
├── requirements.txt
├── .gitignore         # includes input/
├── src/
│   ├── data.py        # LabelMe loading, train/test split
│   ├── flycount.py
│   ├── sex.py         # ♂♀ vs ♀♂
│   ├── orientation.py
│   ├── wingangle.py
│   ├── pipeline.py    # end-to-end per-frame processing
│   ├── train.py       # trains all four models
│   ├── evaluate.py    # Table 1
│   ├── profile.py     # Table 2 + FPS
│   └── demo.py        # live overlay on video (for the demo)
├── models/            # saved models (joblib), gitignored if large
└── results/           # tables and figures
```

## Deliverables and deadlines (from guidelines)

- **Private GitHub repo**, shared with faculty and TAs, with a README covering setup and running.
- **Write-up PDF**: guideline heading says one page, body says two pages — confirm with faculty. Must include problem statement, dataset details, approach, implementation overview, conclusions.
- **Slide deck + live demo** at the review (review window Oct 5–9, 2026); be ready to explain methodology and code structure and to answer Q&A.
- **Submission deadline:** Saturday, October 10, 2026, 11:59 PM.
- Evaluation (10 marks): deliverable quality, code functionality, repo maintenance, write-up clarity, presentation, live demo, Q&A, individual contribution. Keep commits attributable to each team member.
