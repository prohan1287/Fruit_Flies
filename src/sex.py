"""Stage 2 - male-female vs female-male (poster: "MF vs. FM").

Given the two one-fly contours, decide whether the pair is ordered (male, female)
or (female, male). Features: area and moment-based aspect ratio of both contours,
standardized, then logistic regression. Augmented by swapping the pair.
"""

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data import contour_label
from flycount import body_contours
from imgproc import crop_to_contour, well_mask

CATEGORIES = ['mf', 'fm']
FEATURES = ['area1', 'area2', 'aspect1', 'aspect2']


def features(contour_1, contour_2, patch_1, patch_2):
    return np.array([cv2.contourArea(contour_1), cv2.contourArea(contour_2),
                     patch_1.estimate_aspect_ratio(), patch_2.estimate_aspect_ratio()], dtype=float)


def build_dataset(annotations):
    """Two examples per frame with one male and one female contour: (m, f) -> mf and swapped (f, m) -> fm."""
    X, y = [], []
    for anno in tqdm(annotations, desc='Sex data'):
        img = cv2.imread(anno.image_path, cv2.IMREAD_GRAYSCALE)
        found = {'male': [], 'female': []}
        for contour in body_contours(img, well_mask(img)):
            label = contour_label(anno, contour)
            if label in found:
                found[label].append(contour)
        if len(found['male']) != 1 or len(found['female']) != 1:
            continue
        m, f = found['male'][0], found['female'][0]
        pm, pf = crop_to_contour(img, m), crop_to_contour(img, f)
        X.append(features(m, f, pm, pf))
        y.append(CATEGORIES.index('mf'))
        X.append(features(f, m, pf, pm))
        y.append(CATEGORIES.index('fm'))
    return np.array(X), np.array(y)


def make_model():
    # UNSPECIFIED: solver and regularization. Paper describes plain maximum-likelihood
    # (SGD update rule); author's code uses sklearn lbfgs with default C=1.0.
    return make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))


def identify(model, contour_1, contour_2, img):
    """Return {'male': (contour, patch), 'female': (contour, patch)}."""
    p1, p2 = crop_to_contour(img, contour_1), crop_to_contour(img, contour_2)
    label = CATEGORIES[model.predict(features(contour_1, contour_2, p1, p2).reshape(1, -1))[0]]
    if label == 'mf':
        return {'male': (contour_1, p1), 'female': (contour_2, p2)}
    return {'male': (contour_2, p2), 'female': (contour_1, p1)}
