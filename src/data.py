"""LabelMe annotation loading and train/test split.

Annotations live in <data>/images/<subfolder>/*.json (LabelMe format), each
pointing at its frame via `imagePath`. Every shape is a single labelled point:
  fh, fp, fa   female head, body point, abdomen
  mh, mp, ma   male head, body point, abdomen
  mp2          second male body point (wing angles are measured from it)
  mw           male wing tips (zero or two per frame)

<data> is input/ if it exists (author's layout), otherwise the repo root.
"""

import json
import os
from glob import glob

from sklearn.model_selection import train_test_split

from imgproc import in_contour

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'input') if os.path.isdir(os.path.join(ROOT, 'input')) else ROOT
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')
MODELS_DIR = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results')
CACHE_DIR = os.path.join(ROOT, 'cache')

# Paper Sec. 6 says "one third", but every row of Table 1 has N_test = 25% of the examples,
# which is also scikit-learn's default and what the author's code uses (NOTES.md D1).
TEST_SIZE = 0.25
# UNSPECIFIED: random seed. Not given in the sources (author's split is unseeded); team chose 0.
SEED = 0


class Annotation:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.image_path = os.path.join(os.path.dirname(json_path), data['imagePath'])

        # label -> list of (x, y) points
        self.labels = {}
        for shape in data['shapes']:
            self.labels.setdefault(shape['label'], []).append(tuple(shape['points'][0]))

        self.problems = self._check()

    @property
    def name(self):
        return os.path.relpath(self.json_path, IMAGES_DIR)

    def count(self, label):
        return len(self.labels.get(label, []))

    def has(self, label):
        return self.count(label) > 0

    def get(self, label):
        return self.labels[label]

    def _check(self):
        """Return a list of labelling problems (empty if the frame is consistent)."""
        problems = []
        if self.count('fp') != 1 or self.count('mp') != 1:
            problems.append('must have exactly one fp and one mp')
        for h, a in [('fh', 'fa'), ('mh', 'ma')]:
            if self.count(h) != self.count(a) or self.count(h) > 1:
                problems.append('must define both {} and {} or neither'.format(h, a))
        if self.count('mp2') > 1:
            problems.append('at most one mp2')
        if self.count('mw') not in (0, 2):
            problems.append('must define zero or two mw')
        if self.count('mw') == 2 and not all(self.has(k) for k in ('mh', 'ma', 'mp2')):
            problems.append('wings require mh, ma and mp2')
        return problems


def load_annotations(images_dir=IMAGES_DIR):
    """Load every LabelMe JSON file in the subfolders of the images directory, sorted by path."""
    files = sorted(glob(os.path.join(images_dir, '*', '*.json')))
    if not files:
        raise FileNotFoundError('no LabelMe JSON files under {}'.format(images_dir))
    return [Annotation(f) for f in files]


def contour_label(anno, contour):
    """Ground truth for a contour: 'male', 'female', 'both' or 'neither', from the body points."""
    has_m = in_contour(anno.get('mp')[0], contour)
    has_f = in_contour(anno.get('fp')[0], contour)
    if has_m and has_f:
        return 'both'
    if has_m:
        return 'male'
    if has_f:
        return 'female'
    return 'neither'


def split(X, y):
    """Hold out TEST_SIZE of a stage's examples as a test set never used in training.

    As in the author's code, each stage splits its own examples, after augmentation (NOTES.md U3, U4).
    """
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


def summarize(annotations):
    n = len(annotations)
    bad = [a for a in annotations if a.problems]
    print('Annotated frames: {}'.format(n))
    for sub in sorted({os.path.dirname(a.name) for a in annotations}):
        print('  {}: {}'.format(sub, sum(os.path.dirname(a.name) == sub for a in annotations)))
    print('Frames with labelling problems: {}'.format(len(bad)))
    for a in bad:
        print('  {}: {}'.format(a.name, '; '.join(a.problems)))
    for label in ['fh', 'fp', 'fa', 'mh', 'mp', 'mp2', 'ma', 'mw']:
        print('  frames with {:>3}: {}'.format(label, sum(a.has(label) for a in annotations)))
    print('  frames with both wings: {}'.format(sum(a.count('mw') == 2 for a in annotations)))


if __name__ == '__main__':
    summarize(load_annotations())
