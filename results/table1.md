# Table 1: model accuracy

| Model | Train error (ours) | Train error (paper) | Test error (ours) | Test error (paper) | N_train (ours / paper) | N_test (ours / paper) |
|---|---|---|---|---|---|---|
| FlyCount | 0.0% | 0.0% | 0.0% | 0.0% | 759 / 759 | 253 / 253 |
| ♂♀ vs ♀♂ | 0.0% | 0.2% | 0.7% | 0.7% | 423 / 423 | 141 / 141 |
| ♂ orientation | 0.0% | 0.0% | 0.0% | 0.0% | 358 / 358 | 120 / 120 |
| ♀ orientation | 0.8% | 0.8% | 0.0% | 0.0% | 261 / 259 | 87 / 87 |
| WingAngle | σ = 2.35° | σ = 2.06° | σ = 2.18° | σ = 2.92° | 354 / 354 | 118 / 118 |

WingAngle mean test error: 0.011° (paper: 0.592°)
♂♀ vs ♀♂ without aspect ratio: 2.1% test error (poster: 1.4%)

FlyCount decision tree (paper Fig. 4: splits at 1413.5 and 10442.5):
```
|--- area <= 1112.50
|   |--- class: 0
|--- area >  1112.50
|   |--- area <= 10442.50
|   |   |--- class: 1
|   |--- area >  10442.50
|   |   |--- class: 2
```
