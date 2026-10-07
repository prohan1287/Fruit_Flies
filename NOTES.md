# NOTES

Open items, discrepancies between sources, and decisions. Sources: paper (`37.pdf`), poster
(`37-poster.pdf`), guidelines. The author's code (github.com/sgherbst/cs229-project, MIT) is
consulted as a reference where the paper/poster are silent; any value taken from it is marked here.

## Discrepancies

| # | Topic | Paper | Poster | Other evidence | Decision |
|---|---|---|---|---|---|
| D1 | Test split size | Sec. 6: "retaining one third of the dataset for testing" | — | Table 1 (both): N_test is exactly 25% of every row (253/1012, 141/564, 120/478, 87/346, 118/472). Author code uses `train_test_split` default `test_size=0.25`. | **Open** |
| D2 | Throughput ratio | 84 FPS is "2.4x faster" than source | Source is 30 FPS → 2.8x | — | Open |
| D3 | Image size order | 128x64, 96x80 (HxW) | 64x128, 80x96 (WxH) | Author code: HOG `winSize=(64,128)` and `(80,96)` (OpenCV is W,H), so both describe the same images | Same thing; no choice needed |
| D4 | Stage 1 name | FlyCount | Fly/NotFly | — | Use FlyCount |
| D5 | Fig. 4 tree | split at 1413.5 / 10442.5, root value [309, 416, 34] | different (blurry) thresholds and counts | — | Open (paper is legible; poster tree likely from an earlier run) |

## Unspecified in paper/poster

| # | Item | Where | Author code value | Decision |
|---|---|---|---|---|
| U1 | Test fraction (see D1) | data.py | 0.25 | Open |
| U2 | Random seed | data.py | none (unseeded) | Open |
| U3 | Split granularity | data.py | per stage, over examples (contours / augmented pairs), not over frames | Open |
| U4 | Augment before or after split | data.py / stages 2–3 | before split: the swapped or 180°-rotated copy of a training example can land in the test set | Open |

## Data access

- The Dropbox link in the author's README redirects to `*.dl.dropboxusercontent.com`, and that host resets the connection
  from this machine's sandbox, so `input/` has to be downloaded manually for now (see README).
