"""Stage 1 - FlyCount (poster: "Fly/NotFly").

Threshold the frame to keep only fly bodies, extract contours, and classify each
contour as holding 0, 1 or 2 flies from its area with a Gini decision tree.
"""

import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from data import SEED, contour_label
from imgproc import find_contours, threshold_core, well_mask

CATEGORIES = ['neither', 'one', 'both']
FEATURES = ['area']


def body_contours(img, mask):
    return find_contours(threshold_core(img, mask))


def features(contour):
    return np.array([cv2.contourArea(contour)], dtype=float)


def build_dataset(annotations):
    """One example per contour in every annotated frame."""
    X, y = [], []
    for anno in tqdm(annotations, desc='FlyCount data'):
        img = cv2.imread(anno.image_path, cv2.IMREAD_GRAYSCALE)
        for contour in body_contours(img, well_mask(img)):
            label = contour_label(anno, contour)
            X.append(features(contour))
            y.append(CATEGORIES.index('one' if label in ('male', 'female') else label))
    return np.array(X), np.array(y)


def make_model():
    # UNSPECIFIED: tree depth / stopping rule. Paper Fig. 4 shows a fully grown tree; sklearn default
    # (grow until pure) as in the author's code. random_state only fixes tie-breaking.
    return DecisionTreeClassifier(criterion='gini', random_state=SEED)


def classify(model, contours):
    """Group contours by predicted fly count: {'neither': [...], 'one': [...], 'both': [...]}."""
    groups = {c: [] for c in CATEGORIES}
    if len(contours) == 0:
        return groups
    labels = model.predict(np.array([features(c) for c in contours]))
    for contour, label in zip(contours, labels):
        groups[CATEGORIES[label]].append(contour)
    return groups
