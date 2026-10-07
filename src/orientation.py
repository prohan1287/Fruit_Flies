"""Stage 3 - Orientation.

Moment analysis gives the body axis up to 180 degrees. The fly is rotated upright,
cropped to 128x64 (HxW), described by HOG (3780-d), reduced to 15 principal
components, and a logistic regression decides whether to add 180 degrees.
Separate models for male and female flies. Augmented by rotating 180 degrees.
"""

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from data import contour_label
from flycount import body_contours
from imgproc import angle_diff, crop_to_contour, well_mask

CATEGORIES = ['normal', 'flipped']
SEXES = ['male', 'female']
N_COMPONENTS = 15

# UNSPECIFIED: tolerance when deciding from the head/abdomen labels whether the moment angle is
# flipped; frames outside it are dropped. Author code: 0.1 rad.
LABEL_TOL = 0.1


def make_hog():
    # Paper/poster give 8x8 cells and 9 bins only.
    # UNSPECIFIED: block 16x16, block stride 8x8 (author code; OpenCV/Dalal-Triggs defaults).
    # Window is (W, H) = (64, 128), giving the paper's 3780-d descriptor.
    return cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)


def hog_patch(patch, angle=None):
    """Rotate the body upright and crop/resample to 128x64.

    UNSPECIFIED: crop/resample method. Author code: centre a 256x128 window on the body's centre
    of mass, then keep every second pixel.
    """
    return patch.orient_vertical(angle).recenter(128, 256).downsample(2)


def hog_features(hog, patch):
    return hog.compute(patch.img).flatten()


def label_angle(head, abdomen):
    """Body direction (abdomen -> head) in the moment-angle convention (y up)."""
    return np.arctan2(abdomen[1] - head[1], head[0] - abdomen[0])


def build_dataset(annotations):
    """Per sex: HOG of the upright fly, label normal/flipped, plus the 180-rotated copy with the opposite label."""
    hog = make_hog()
    X = {s: [] for s in SEXES}
    y = {s: [] for s in SEXES}
    keys = {'male': ('mh', 'ma'), 'female': ('fh', 'fa')}
    dropped = 0
    for anno in tqdm(annotations, desc='Orientation data'):
        img = cv2.imread(anno.image_path, cv2.IMREAD_GRAYSCALE)
        for contour in body_contours(img, well_mask(img)):
            sex = contour_label(anno, contour)
            if sex not in SEXES or not all(anno.has(k) for k in keys[sex]):
                continue
            head, abdomen = (anno.get(k)[0] for k in keys[sex])
            patch = crop_to_contour(img, contour)

            diff = abs(angle_diff(patch.estimate_angle(), label_angle(head, abdomen)))
            if diff <= LABEL_TOL:
                label = 0
            elif np.pi - LABEL_TOL <= diff <= np.pi + LABEL_TOL:
                label = 1
            else:
                dropped += 1
                continue

            upright = hog_patch(patch)
            X[sex].append(hog_features(hog, upright))
            y[sex].append(label)
            X[sex].append(hog_features(hog, upright.rotate180()))
            y[sex].append(1 - label)
    print('Orientation: dropped {} flies whose moment angle disagreed with the labels by more than {} rad'
          .format(dropped, LABEL_TOL))
    return {s: np.array(X[s]) for s in SEXES}, {s: np.array(y[s]) for s in SEXES}


def make_model():
    # UNSPECIFIED: logistic regression solver/regularization. Author code: lbfgs, C=1.0.
    return make_pipeline(PCA(n_components=N_COMPONENTS), LogisticRegression(solver='lbfgs'))


class OrientationPredictor:
    def __init__(self, model):
        self.model = model
        self.hog = make_hog()

    def predict(self, patch):
        """Return (centre (x, y) in frame coordinates, 0-360 body angle in radians)."""
        center = patch.estimate_center(absolute=True)
        angle = patch.estimate_angle()
        features = hog_features(self.hog, hog_patch(patch, angle)).reshape(1, -1)
        if CATEGORIES[self.model.predict(features)[0]] == 'flipped':
            angle += np.pi
        return center, angle
