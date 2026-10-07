"""Reproduce Table 1 (model accuracy) next to the paper's numbers.

Run src/train.py first. Writes results/table1.md.
"""

import os

import joblib
import numpy as np
from sklearn.tree import export_text

import flycount
import sex
from data import RESULTS_DIR
from train import SPLITS_FILE, load_models

ROWS = [('flycount', 'FlyCount'), ('sex', '♂♀ vs ♀♂'), ('orient_male', '♂ orientation'),
        ('orient_female', '♀ orientation'), ('wingangle', 'WingAngle')]

PAPER = {
    'flycount': ('0.0%', '0.0%', 759, 253),
    'sex': ('0.2%', '0.7%', 423, 141),
    'orient_male': ('0.0%', '0.0%', 358, 120),
    'orient_female': ('0.8%', '0.0%', 259, 87),
    'wingangle': ('σ = 2.06°', 'σ = 2.92°', 354, 118),
}


def error(model, X, y, regression):
    if regression:
        return 'σ = {:.2f}°'.format(np.std(y - model.predict(X)))
    return '{:.1f}%'.format(100 * np.mean(model.predict(X) != y))


def table1(models, splits):
    lines = ['| Model | Train error (ours) | Train error (paper) | Test error (ours) | Test error (paper) '
             '| N_train (ours / paper) | N_test (ours / paper) |',
             '|---|---|---|---|---|---|---|']
    for stage, name in ROWS:
        X_train, X_test, y_train, y_test = splits[stage]
        reg = stage == 'wingangle'
        p = PAPER[stage]
        lines.append('| {} | {} | {} | {} | {} | {} / {} | {} / {} |'.format(
            name, error(models[stage], X_train, y_train, reg), p[0],
            error(models[stage], X_test, y_test, reg), p[1], len(y_train), p[2], len(y_test), p[3]))
    return lines


def main():
    models = load_models()
    splits = joblib.load(SPLITS_FILE)
    out = ['# Table 1: model accuracy', ''] + table1(models, splits)

    # WingAngle mean test error (paper: 0.592°)
    X_train, X_test, y_train, y_test = splits['wingangle']
    mean_err = np.mean(y_test - models['wingangle'].predict(X_test))
    out += ['', 'WingAngle mean test error: {:.3f}° (paper: 0.592°)'.format(mean_err)]

    # Poster ablation: ♂♀ vs ♀♂ with the aspect-ratio features removed (paper: 1.4% test error)
    X_train, X_test, y_train, y_test = splits['sex']
    area = [sex.FEATURES.index('area1'), sex.FEATURES.index('area2')]
    ablation = sex.make_model().fit(X_train[:, area], y_train)
    err = 100 * np.mean(ablation.predict(X_test[:, area]) != y_test)
    out += ['♂♀ vs ♀♂ without aspect ratio: {:.1f}% test error (poster: 1.4%)'.format(err)]

    # Fig. 4 decision tree (paper: area <= 1413.5 -> neither; area <= 10442.5 -> one; else both)
    out += ['', 'FlyCount decision tree (paper Fig. 4: splits at 1413.5 and 10442.5):', '```',
            export_text(models['flycount'], feature_names=flycount.FEATURES).rstrip(), '```']

    text = '\n'.join(out)
    print(text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'table1.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
