"""End-to-end per-frame processing: FlyCount -> ♂♀ vs ♀♂ -> Orientation -> WingAngle."""

from collections import defaultdict
from time import perf_counter

import numpy as np

import flycount
import sex
from imgproc import crop_to_contour, well_mask
from orientation import OrientationPredictor
from train import load_models
from wingangle import WingPredictor, male_fly_patch

STAGE_NAMES = ['I/O', 'FlyCount', '♂♀ vs ♀♂', 'Orientation', 'WingAngle']


class Profiler:
    """Accumulates wall-clock time per named step."""

    def __init__(self):
        self.times = defaultdict(list)
        self._start = {}

    def tick(self, name):
        self._start[name] = perf_counter()

    def tock(self, name):
        self.times[name].append(perf_counter() - self._start.pop(name))

    def mean_ms(self, name):
        return 1e3 * np.mean(self.times[name]) if self.times[name] else 0.0


class _NullProfiler:
    def tick(self, name):
        pass

    def tock(self, name):
        pass


class Pipeline:
    def __init__(self, models=None):
        models = models or load_models()
        self.flycount = models['flycount']
        self.sex = models['sex']
        self.orientation = {'male': OrientationPredictor(models['orient_male']),
                            'female': OrientationPredictor(models['orient_female'])}
        self.wings = WingPredictor(models['wingangle'])
        self.mask = None

    def process(self, img, prof=None):
        """Analyse one grayscale frame.

        Returns a dict keyed by 'male'/'female' (contour, center, angle, axes; male also
        wing_right/wing_left in degrees), or by 'both' when the flies form one contour.
        Empty when the frame is not one of those two cases.
        """
        prof = prof or _NullProfiler()
        if self.mask is None or self.mask.shape != img.shape:
            self.mask = well_mask(img)

        prof.tick('FlyCount')
        groups = flycount.classify(self.flycount, flycount.body_contours(img, self.mask))
        prof.tock('FlyCount')

        if len(groups['one']) == 0 and len(groups['both']) == 1:
            contour = groups['both'][0]
            return {'both': {'contour': contour, 'center': crop_to_contour(img, contour).estimate_center()}}
        if not (len(groups['one']) == 2 and len(groups['both']) == 0):
            return {}

        prof.tick('♂♀ vs ♀♂')
        flies = sex.identify(self.sex, groups['one'][0], groups['one'][1], img)
        prof.tock('♂♀ vs ♀♂')

        prof.tick('Orientation')
        results = {}
        for s, (contour, patch) in flies.items():
            center, angle = self.orientation[s].predict(patch)
            results[s] = {'contour': contour, 'center': center, 'angle': angle,
                          'axes': patch.estimate_axes()}
        prof.tock('Orientation')

        prof.tick('WingAngle')
        male = results['male']
        fly = male_fly_patch(img, self.mask, male['center'], male['angle'])
        if fly is not None:
            male['wing_right'], male['wing_left'] = self.wings.predict(fly)
        prof.tock('WingAngle')

        return results
