"""Train all four pipeline models and save them with joblib.

Each stage builds its examples from the annotated frames (with that stage's
augmentation), holds out a test split (data.split), and fits on the rest.
Feature matrices are cached in cache/ and the train/test splits are saved
there too, for evaluate.py and figures.py.

    python src/train.py            # rebuild features, train, save
    python src/train.py --cached   # reuse cache/features.joblib
"""

import os
from argparse import ArgumentParser

import joblib

import flycount
import orientation
import sex
import wingangle
from data import CACHE_DIR, MODELS_DIR, load_annotations, split

FEATURES_FILE = os.path.join(CACHE_DIR, 'features.joblib')
SPLITS_FILE = os.path.join(CACHE_DIR, 'splits.joblib')

# stage name -> (module, saved model file)
STAGES = {
    'flycount': (flycount, 'flycount.joblib'),
    'sex': (sex, 'sex.joblib'),
    'orient_male': (orientation, 'orientation_male.joblib'),
    'orient_female': (orientation, 'orientation_female.joblib'),
    'wingangle': (wingangle, 'wingangle.joblib'),
}


def build_features():
    annotations = load_annotations()
    X_or, y_or = orientation.build_dataset(annotations)
    return {
        'flycount': flycount.build_dataset(annotations),
        'sex': sex.build_dataset(annotations),
        'orient_male': (X_or['male'], y_or['male']),
        'orient_female': (X_or['female'], y_or['female']),
        'wingangle': wingangle.build_dataset(annotations),
    }


def model_path(stage):
    return os.path.join(MODELS_DIR, STAGES[stage][1])


def load_models():
    return {stage: joblib.load(model_path(stage)) for stage in STAGES}


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cached', action='store_true', help='reuse cached feature matrices')
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.cached and os.path.exists(FEATURES_FILE):
        features = joblib.load(FEATURES_FILE)
    else:
        features = build_features()
        joblib.dump(features, FEATURES_FILE)

    splits = {}
    for stage, (module, _) in STAGES.items():
        X, y = features[stage]
        X_train, X_test, y_train, y_test = split(X, y)
        model = module.make_model().fit(X_train, y_train)
        joblib.dump(model, model_path(stage))
        splits[stage] = (X_train, X_test, y_train, y_test)
        print('{:<14} trained on {:>4} examples, {:>4} held out -> {}'.format(
            stage, len(y_train), len(y_test), os.path.relpath(model_path(stage))))
    joblib.dump(splits, SPLITS_FILE)


if __name__ == '__main__':
    main()
