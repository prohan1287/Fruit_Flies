"""Image-processing helpers shared by all pipeline stages.

Thresholds, kernel sizes and crop geometry are not given in the paper or poster;
the values here come from the author's code (github.com/sgherbst/cs229-project,
MIT). Each one is marked UNSPECIFIED and listed in NOTES.md.
"""

from math import degrees

import cv2
import imutils
import numpy as np

# UNSPECIFIED: Stage 1 body threshold. Author code: fixed intensity 115 (flies are dark).
CORE_THRESHOLD = 115
# UNSPECIFIED: Stage 4 median blur kernel. Author code: 5.
WING_BLUR_KSIZE = 5
# UNSPECIFIED: Stage 4 threshold. Author code: (mean intensity inside the well) - 5.
WING_MEAN_OFFSET = 5
# UNSPECIFIED: Stage 4 erosion. Author code: 9x9 ellipse, 1 iteration.
WING_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
WING_ERODE_ITERATIONS = 1


def well_mask(img):
    """Binary mask of the circular well that exactly fills the square frame."""
    rows, cols = img.shape[:2]
    assert rows == cols
    r = (rows - 1) // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    return mask


def bound_point(pt, img):
    """Round (x, y) to integer pixel coordinates inside img."""
    x = int(np.clip(np.round(pt[0]), 0, img.shape[1] - 1))
    y = int(np.clip(np.round(pt[1]), 0, img.shape[0] - 1))
    return x, y


def angle_diff(a, b):
    """Smallest signed difference a - b, in radians, wrapped to [-pi, pi)."""
    return ((a - b) + np.pi) % (2 * np.pi) - np.pi


def threshold_core(img, mask):
    """Stage 1: keep only the dark central body of each fly (drops background, legs, wings)."""
    _, bw = cv2.threshold(img, CORE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(bw, mask)


def threshold_wings(img, mask):
    """Stage 4: median blur, threshold relative to the well's mean intensity, erode.

    Keeps the translucent wings attached to the body while the erosion removes the thin legs.
    """
    blurred = cv2.medianBlur(img, WING_BLUR_KSIZE)
    mean_in_well = np.sum((mask == 255) * img) / np.sum(mask == 255)
    _, bw = cv2.threshold(blurred, mean_in_well - WING_MEAN_OFFSET, 255, cv2.THRESH_BINARY_INV)
    eroded = cv2.erode(bw, WING_ERODE_KERNEL, iterations=WING_ERODE_ITERATIONS)
    return cv2.bitwise_and(eroded, mask)


def find_contours(bw):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def in_contour(pt, contour):
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) > 0


def contour_roi(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return slice(y, y + h), slice(x, x + w)


def mask_from_contour(shape, contour):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask


def crop_to_contour(img, contour):
    """ImagePatch holding only the pixels inside the contour, cropped to its bounding box."""
    roi = contour_roi(contour)
    mask = mask_from_contour(img.shape, contour)[roi]
    crop = cv2.bitwise_and(img[roi], img[roi], mask=mask)
    return ImagePatch(crop, mask, ulx=roi[1].start, uly=roi[0].start)


class ImagePatch:
    """A grayscale crop plus its mask and its upper-left corner in the full frame."""

    def __init__(self, img, mask=None, ulx=0, uly=0):
        self.img = img
        self.mask = mask
        self.ulx = ulx
        self.uly = uly
        self._moments = None

    @property
    def moments(self):
        if self._moments is None:
            self._moments = cv2.moments(self.img)
        return self._moments

    def estimate_angle(self):
        """Body axis from central moments: theta = 1/2 atan(2 mu11 / (mu20 - mu02)).

        Has a 0/180 degree ambiguity. Angles are counter-clockwise from +x with y pointing up.
        """
        m = self.moments
        angle = 0.5 * np.arctan(2 * m['mu11'] / (m['mu20'] - m['mu02']))
        if m['mu20'] < m['mu02']:
            angle += np.pi / 2
        return -angle

    def estimate_center(self, absolute=True):
        m = self.moments
        x, y = m['m10'] / m['m00'], m['m01'] / m['m00']
        if absolute:
            x, y = x + self.ulx, y + self.uly
        return x, y

    def estimate_axes(self):
        """Major and minor axis lengths of the equivalent ellipse."""
        m = self.moments
        mu20, mu02, mu11 = m['mu20'] / m['m00'], m['mu02'] / m['m00'], m['mu11'] / m['m00']
        root = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
        major = np.sqrt(6 * (mu20 + mu02 + root))
        minor = np.sqrt(6 * (mu20 + mu02 - root))
        return major, minor

    def estimate_aspect_ratio(self):
        major, minor = self.estimate_axes()
        return major / minor

    def rotate(self, angle):
        """Rotate by angle (radians, counter-clockwise), growing the canvas to fit."""
        img = imutils.rotate_bound(self.img, degrees(angle))
        mask = imutils.rotate_bound(self.mask, degrees(angle)) if self.mask is not None else None
        return ImagePatch(img, mask)

    def rotate180(self):
        return self.rotate(np.pi)

    def orient_vertical(self, angle=None):
        """Rotate so the body axis (at `angle`, default from moments) points up."""
        if angle is None:
            angle = self.estimate_angle()
        return self.rotate(angle - np.pi / 2)

    def recenter(self, width, height, center=None):
        """Crop/pad to width x height with `center` (default: centre of mass) in the middle."""
        if center is None:
            center = bound_point(self.estimate_center(absolute=False), self.img)
        ox, oy = center
        nx, ny = width // 2, height // 2

        left, up = min(ox, nx), min(oy, ny)
        right = min(width - nx, self.img.shape[1] - ox)
        down = min(height - ny, self.img.shape[0] - oy)
        new_win = (slice(ny - up, ny + down), slice(nx - left, nx + right))
        old_win = (slice(oy - up, oy + down), slice(ox - left, ox + right))

        img = np.zeros((height, width), dtype=np.uint8)
        img[new_win] = self.img[old_win]
        mask = None
        if self.mask is not None:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[new_win] = self.mask[old_win]
        return ImagePatch(img, mask, ulx=self.ulx + ox - nx, uly=self.uly + oy - ny)

    def downsample(self, factor):
        mask = self.mask[::factor, ::factor] if self.mask is not None else None
        return ImagePatch(self.img[::factor, ::factor], mask)

    def flip_horizontal(self):
        mask = cv2.flip(self.mask, 1) if self.mask is not None else None
        return ImagePatch(cv2.flip(self.img, 1), mask)
