# NOTES

Open items, discrepancies between sources, and decisions. Sources: paper (`37.pdf`), poster
(`37-poster.pdf`), guidelines. Where the paper and poster are silent, values come from the author's
code (github.com/sgherbst/cs229-project, MIT); each is marked `# UNSPECIFIED:` in `src/`.
Team decision: follow the paper faithfully, using the author's code to fill gaps.

## Discrepancies

| # | Topic | Paper | Poster | Other evidence | Decision |
|---|---|---|---|---|---|
| D1 | Test split size | Sec. 6: "retaining one third of the dataset for testing" | — | Table 1: N_test is exactly 25% of every row (253/1012, 141/564, 120/478, 87/346, 118/472). Author code: `train_test_split` default `test_size=0.25`. | 25%. Gives Table 1's N exactly. |
| D2 | Throughput ratio | 84 FPS is "2.4x faster" than source | Source is 30 FPS → 2.8x | The test videos report 35 FPS in their container. 84/35 = 2.4, which matches the paper's figure. | Report both ratios. profile.py uses 30 (poster). |
| D3 | Image size order | 128x64, 96x80 (HxW) | 64x128, 80x96 (WxH) | Author HOG `winSize=(64,128)`, `(80,96)` (OpenCV is W,H) | Same images. HOG sizes match exactly (3780, 3564). |
| D4 | Stage 1 name | FlyCount | Fly/NotFly | — | FlyCount |
| D5 | Fig. 4 tree | splits at 1413.5 / 10442.5, root value [309, 416, 34] | different, blurry thresholds and counts | — | Ours: 1112.5 / 10442.5. The lower split is the midpoint between the largest "neither" and smallest "one" contour in the training split, so it moves with the split. |
| D6 | Stage 4 threshold | "adaptive thresholding" | "threshold" | Author code: global threshold at (mean intensity of the well − 5), not `cv2.adaptiveThreshold`. "Adaptive" means adapted to each frame's mean. | Follow author code |
| D7 | Logistic regression fitting | Shows the SGD update rule θ := θ + α(y − h)x | same | Author code: sklearn `LogisticRegression(solver='lbfgs')`, default L2 with C=1.0 | Follow author code. Both maximise the same likelihood; only the L2 term differs. |
| D8 | Orientation augmentation | Footnote 2 says "for this regression", but it sits on the orientation classifier | "Data augmented by rotating examples 180°" | Author code: augmentation in orientation loader | Orientation stage |
| D9 | Wing data | "Repeat for the left wing" | same | Author code: one shared regressor. The left wing is the right-wing crop of the horizontally flipped fly. Each male gives 2 examples (354+118 = 2 × 236). | Follow author code |

## Unspecified in paper/poster (values used)

| # | Item | Where | Value (source) |
|---|---|---|---|
| U1 | Test fraction | data.py | 0.25 (Table 1 / author) |
| U2 | Random seed | data.py | 0 (team choice; author's split is unseeded) |
| U3 | Split granularity | data.py | per stage, over that stage's examples (author) |
| U4 | Augment before or after split | data.py | before (author). The swapped/rotated twin of a training example can land in the test set, which is optimistic for test error. |
| U5 | Stage 1 threshold | imgproc.py | fixed 115, binary inverse, inside circular well mask (author) |
| U6 | Stage 4 median blur | imgproc.py | 5 (author) |
| U7 | Stage 4 threshold | imgproc.py | well mean − 5 (author) |
| U8 | Stage 4 erosion | imgproc.py | 9x9 ellipse, 1 iteration (author) |
| U9 | HOG block / stride | orientation.py, wingangle.py | 16x16 block, 8x8 stride (author; Dalal-Triggs default) |
| U10 | Decision tree depth | flycount.py | unlimited, sklearn default (author) |
| U11 | Logistic regression | sex.py, orientation.py | lbfgs, C=1.0 (author) |
| U12 | Orientation label tolerance | orientation.py | 0.1 rad. Flies whose moment axis disagrees with the head/abdomen labels by more are dropped (author; 21 dropped). |
| U13 | Orientation crop | orientation.py | 256x128 window around the centre of mass, every 2nd pixel → 128x64 (author) |
| U14 | Wing ROI and crop | wingangle.py | 400x400 ROI; upright rows [−68, +124), cols [0, +160) from centre, every 2nd pixel → 96x80 (author) |
| U15 | Wing labels | wingangle.py | wing tips rotated into body frame, measured from mp2, atan(\|dx\|/\|dy\|) (author) |
| U16 | Linear regression intercept | wingangle.py | Closed form θ = (XᵀX)⁻¹Xᵀy with a bias column (CS229 convention x₀ = 1). Author used sklearn LinearRegression; the results are identical. |
| U17 | Fig. 10 video | figures.py | test4.mp4 (author's demo default). Note: annotated folder 12-08_11-15-00 was taken from test4, so some training frames come from this clip. |
| U18 | Fig. 10 correlation | figures.py | Pearson of the plotted curves (right, −left): ours −0.68 vs paper −0.64 |

## Environment deviations

- OpenCV 5 removed `cv2.HOGDescriptor`, so opencv-python is pinned to 4.14.
- Data was supplied at `images/` and `videos/` in the repo root, not `input/`. `data.py` uses `input/` if present, else the root. Both are gitignored.
- The female orientation N_train is 261 vs the paper's 259 (one extra labelled female passes the 0.1 rad check). This is probably OpenCV/imutils version differences in the moment angle.
- Timing varies a lot on this laptop (Balanced power plan, background apps), and the first pass is a cold start. profile.py reports the median of 5 passes and lists each one.
