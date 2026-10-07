"""Stage 4 - WingAngle (male only).

A region around the male is segmented with median blur, threshold and erosion
(keeps wings, drops legs) and rotated upright using Stage 3. The right half is
cropped to 96x80 (HxW), described by HOG (3564-d), reduced to 40 principal
components, and a closed-form linear regression gives the 0-90 degree wing angle
from the body's major axis. The left wing is the same after a horizontal flip.
"""

import cv2
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from data import SEED, contour_label
from flycount import body_contours
from imgproc import (ImagePatch, angle_diff, bound_point, crop_to_contour, find_contours, in_contour,
                     mask_from_contour, threshold_wings, well_mask)
from orientation import LABEL_TOL, label_angle

N_COMPONENTS = 40

# UNSPECIFIED: ROI around the male before segmentation. Author code: 400x400 centred on the body.
ROI_SIZE = 400
# UNSPECIFIED: half-image crop. Author code: rows [cy-68, cy+124), cols [cx, cx+160) of the upright
# ROI, then every second pixel -> 96x80.
CROP_ROWS = (-68, 124)
CROP_COLS = (0, 160)


def make_hog():
    # UNSPECIFIED: block 16x16, block stride 8x8 (author code). Window (W, H) = (80, 96) -> 3564-d.
    return cv2.HOGDescriptor((80, 96), (16, 16), (8, 8), (8, 8), 9)


def male_fly_patch(img, mask, body_center, body_angle):
    """Segment the male with its wings and rotate it upright. None if segmentation fails."""
    center = bound_point(body_center, img)
    roi = ImagePatch(img, mask).recenter(ROI_SIZE, ROI_SIZE, center)

    # keep only the smallest wing-threshold contour containing the body centre
    mid = (ROI_SIZE // 2, ROI_SIZE // 2)
    for contour in sorted(find_contours(threshold_wings(roi.img, roi.mask)), key=len):
        if in_contour(mid, contour):
            roi.mask = cv2.bitwise_and(roi.mask, mask_from_contour(roi.img.shape, contour))
            roi.img = cv2.bitwise_and(roi.img, roi.img, mask=roi.mask)
            break
    else:
        return None
    return roi.orient_vertical(body_angle)


def hog_patch(fly_patch):
    """Right half of the upright male, 96x80."""
    cy, cx = fly_patch.img.shape[0] // 2, fly_patch.img.shape[1] // 2
    window = (slice(cy + CROP_ROWS[0], cy + CROP_ROWS[1]), slice(cx + CROP_COLS[0], cx + CROP_COLS[1]))
    return ImagePatch(fly_patch.img[window], fly_patch.mask[window]).downsample(2)


def hog_features(hog, patch):
    return hog.compute(patch.img).flatten()


def wing_labels(anno, origin, body_angle):
    """Right and left wing angles (degrees) from the labels.

    UNSPECIFIED: how labels are computed from the points. Author code: rotate the wing tips (mw)
    into the upright body frame, measure each from mp2, angle = atan(|dx| / |dy|) from the body axis.
    The wing on the +x side of the upright fly is the right wing.
    """
    c, s = np.cos(np.pi / 2 - body_angle), np.sin(np.pi / 2 - body_angle)
    rot = np.array([[c, -s], [s, c]])

    def upright(pt):  # frame pixel -> upright body frame, y up
        return rot.dot([pt[0] - origin[0], origin[1] - pt[1]])

    base = upright(anno.get('mp2')[0])
    wings = []
    for tip in anno.get('mw'):
        dx, dy = upright(tip) - base
        wings.append((dx, np.degrees(np.arctan(abs(dx) / abs(dy)))))
    wings.sort()
    return wings[1][1], wings[0][1]


def build_dataset(annotations):
    """Two examples per labelled male: right half, and left half (horizontally flipped)."""
    hog = make_hog()
    X, y = [], []
    dropped = 0
    for anno in tqdm(annotations, desc='WingAngle data'):
        if not (anno.count('mw') == 2 and all(anno.has(k) for k in ('mh', 'ma', 'mp2'))):
            continue
        img = cv2.imread(anno.image_path, cv2.IMREAD_GRAYSCALE)
        mask = well_mask(img)
        male = next((c for c in body_contours(img, mask) if contour_label(anno, c) == 'male'), None)
        if male is None:
            continue

        # true body angle: moment angle, corrected by 180 degrees using the head/abdomen labels
        body = crop_to_contour(img, male)
        angle = body.estimate_angle()
        diff = abs(angle_diff(angle, label_angle(anno.get('mh')[0], anno.get('ma')[0])))
        if np.pi - LABEL_TOL <= diff <= np.pi + LABEL_TOL:
            angle += np.pi
        elif diff > LABEL_TOL:
            dropped += 1
            continue

        center = body.estimate_center(absolute=True)
        fly = male_fly_patch(img, mask, center, angle)
        if fly is None:
            dropped += 1
            continue

        right, left = wing_labels(anno, bound_point(center, img), angle)
        X.append(hog_features(hog, hog_patch(fly)))
        y.append(right)
        X.append(hog_features(hog, hog_patch(fly.flip_horizontal())))
        y.append(left)
    print('WingAngle: dropped {} labelled males (angle/segmentation failure)'.format(dropped))
    return np.array(X), np.array(y)


class ClosedFormLinearRegression(BaseEstimator, RegressorMixin):
    """Least squares by the normal equation theta = (X^T X)^-1 X^T y (paper Sec. 5.1), with an intercept term."""

    def fit(self, X, y):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        self.theta_ = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
        return self

    def predict(self, X):
        return self.theta_[0] + X @ self.theta_[1:]


def make_model():
    return make_pipeline(PCA(n_components=N_COMPONENTS, random_state=SEED), ClosedFormLinearRegression())


class WingPredictor:
    def __init__(self, model):
        self.model = model
        self.hog = make_hog()

    def predict(self, fly_patch):
        """Right and left wing angles in degrees."""
        right = hog_features(self.hog, hog_patch(fly_patch))
        left = hog_features(self.hog, hog_patch(fly_patch.flip_horizontal()))
        return self.model.predict(np.vstack([right, left]))
