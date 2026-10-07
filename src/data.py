"""LabelMe annotation loading and train/test split.

Annotations live in input/images/<subfolder>/*.json (LabelMe format), each
pointing at its frame via `imagePath`. Every shape is a single labelled point:
  fh, fp, fa   female head, body point, abdomen
  mh, mp, ma   male head, body point, abdomen
  mp2          second male body point (needed for wing labels)
  mw           male wing tips (zero or two per frame)
"""

import json
import os
from glob import glob

from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, 'input', 'images')


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
    """Load every LabelMe JSON file in the subfolders of input/images, sorted by path."""
    files = sorted(glob(os.path.join(images_dir, '*', '*.json')))
    if not files:
        raise FileNotFoundError('no LabelMe JSON files under {}'.format(images_dir))
    return [Annotation(f) for f in files]


def split(items, test_size, seed):
    """Hold out a test set that is never used in training.

    # UNSPECIFIED: test fraction. Paper Section 6 says "one third of the dataset",
    # but every row of Table 1 has N_test/(N_train+N_test) = 25.0% (see NOTES.md).
    # UNSPECIFIED: random seed. Not given in the sources.
    # UNSPECIFIED: split granularity (whole frames vs. per-stage examples). See NOTES.md.
    """
    return train_test_split(items, test_size=test_size, random_state=seed)


def summarize(annotations):
    n = len(annotations)
    bad = [a for a in annotations if a.problems]
    print('Annotated frames: {}'.format(n))
    print('Frames with labelling problems: {}'.format(len(bad)))
    for a in bad:
        print('  {}: {}'.format(os.path.relpath(a.json_path, ROOT), '; '.join(a.problems)))
    for label in ['fh', 'fp', 'fa', 'mh', 'mp', 'mp2', 'ma', 'mw']:
        print('  frames with {:>3}: {}'.format(label, sum(a.has(label) for a in annotations)))
    print('  frames with both wings: {}'.format(sum(a.count('mw') == 2 for a in annotations)))


if __name__ == '__main__':
    summarize(load_annotations())
