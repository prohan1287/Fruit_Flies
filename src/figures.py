"""Reproduce the paper's figures into results/.

  fig4_tree.png                FlyCount decision tree (paper Fig. 4)
  fig8_orientation_pca.png     PC1 vs PC2 of female orientation HOG, normal vs flipped (Fig. 8)
  fig9_wing_pc1.png            Wing angle vs PC1 of wing HOG (Fig. 9)
  fig10_wing_time.png          Male wing angles over time on a test video (Fig. 10)
  poster_explained_variance.png  Cumulative explained variance, orientation and wing PCA (poster)

Run src/train.py first.
"""

import os
from argparse import ArgumentParser

import cv2
import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

import flycount  # noqa: E402
import orientation  # noqa: E402
from data import RESULTS_DIR, VIDEOS_DIR  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from train import SPLITS_FILE, load_models  # noqa: E402

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'lines.linewidth': 2,
    'legend.frameon': False,
})


def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.relpath(path))


def fig4(models):
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_tree(models['flycount'], feature_names=flycount.FEATURES, class_names=flycount.CATEGORIES,
              filled=True, rounded=True, impurity=True, ax=ax)
    ax.set_title('FlyCount decision tree (paper Fig. 4: splits at 1413.5 and 10442.5)', loc='left')
    save(fig, 'fig4_tree.png')


def fig8(splits):
    X_train, _, y_train, _ = splits['orient_female']
    Z = PCA(n_components=2).fit_transform(X_train)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for label, color, marker in [(0, BLUE, 'o'), (1, ORANGE, 'x')]:
        pts = Z[y_train == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color=color, marker=marker, alpha=0.8,
                   linewidths=1.5 if marker == 'x' else 0, label=orientation.CATEGORIES[label])
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Female orientation HOG, first two principal components', loc='left')
    ax.legend(loc='upper right')
    save(fig, 'fig8_orientation_pca.png')


def fig9(splits):
    X_train, _, y_train, _ = splits['wingangle']
    pc1 = PCA(n_components=1).fit_transform(X_train)[:, 0]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(pc1, y_train, s=20, color=BLUE, alpha=0.7, linewidths=0)
    ax.set_xlabel('PCA component 1')
    ax.set_ylabel('Wing angle (deg)')
    ax.set_title('Wing angle vs first principal component of wing HOG', loc='left')
    save(fig, 'fig9_wing_pc1.png')


def explained_variance(splits):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    curves = [('orient_male', 'Male orientation', orientation.N_COMPONENTS, BLUE, '-', 0.42),
              ('orient_female', 'Female orientation', orientation.N_COMPONENTS, BLUE, '--', 0.50),
              ('wingangle', 'Wing angle', 40, ORANGE, '-', 0.58)]
    for stage, name, chosen, color, style, label_y in curves:
        X_train = splits[stage][0]
        cum = np.cumsum(PCA(n_components=200).fit(X_train).explained_variance_ratio_)
        ax.plot(np.arange(1, 201), cum, color=color, linestyle=style, label=name)
        ax.plot(chosen, cum[chosen - 1], 'o', color=color, markersize=8, markeredgecolor='#fcfcfb',
                markeredgewidth=2)
        ax.annotate('{}: {} PCs, {:.0%}'.format(name, chosen, cum[chosen - 1]), (chosen, cum[chosen - 1]),
                    xytext=(60, label_y), color=INK_2, fontsize=9, va='center',
                    arrowprops=dict(arrowstyle='-', color=INK_2, linewidth=0.6, shrinkB=5))
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Total explained variance ratio')
    ax.set_ylim(0, 1.02)
    ax.set_title('Cumulative explained variance (poster)', loc='left')
    ax.legend(loc='lower right')
    save(fig, 'poster_explained_variance.png')


def fig10(models, video):
    pipeline = Pipeline(models)
    cap = cv2.VideoCapture(os.path.join(VIDEOS_DIR, video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    t, right, left = [], [], []
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        male = pipeline.process(frame[:, :, 0]).get('male', {})
        t.append(frame_no / fps)
        # frames without a wing estimate stay as gaps in the plot
        right.append(male.get('wing_right', np.nan))
        left.append(male.get('wing_left', np.nan))
        frame_no += 1
    right, left = np.array(right), np.array(left)

    # Paper plots the left wing as a negative angle; the -0.64 it reports is not defined further.
    ok = ~np.isnan(right)
    r_raw = np.corrcoef(right[ok], left[ok])[0, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, right, color=BLUE, linewidth=1.2, label='Right wing')
    ax.plot(t, -left, color=ORANGE, linewidth=1.2, label='Left wing')
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wing angle (degrees)')
    ax.set_title('Male wing angle over time, {} ({} of {} frames analysed)'.format(video, ok.sum(), frame_no),
                 loc='left')
    ax.legend(loc='upper left')
    save(fig, 'fig10_wing_time.png')
    print('Fig. 10 correlation of right and left wing angles: {:.2f} '
          '(of the plotted right and -left curves: {:.2f}; paper: -0.64)'.format(r_raw, -r_raw))
    return r_raw


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-i', '--input', default='test4.mp4', help='video for Fig. 10')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    models = load_models()
    splits = joblib.load(SPLITS_FILE)
    fig4(models)
    fig8(splits)
    fig9(splits)
    explained_variance(splits)
    r = fig10(models, args.input)
    with open(os.path.join(RESULTS_DIR, 'fig10_correlation.txt'), 'w') as f:
        f.write('video={} pearson(right, left)={:.3f} pearson(right, -left)={:.3f} paper=-0.64\n'
                .format(args.input, r, -r))


if __name__ == '__main__':
    main()
