"""Live overlay of the pipeline's output on a test video (for the review demo).

Female: blue outline and heading arrow. Male: red outline and heading arrow, plus
cyan arrows for the right and left wing angles. Magenta: flies in contact.

    python src/demo.py -i test4.mp4                  # live window, q to quit, space to pause
    python src/demo.py -i test4.mp4 --no-display --write-video
"""

import os
from argparse import ArgumentParser
from time import perf_counter

import cv2
import numpy as np

from data import RESULTS_DIR, VIDEOS_DIR
from imgproc import bound_point
from pipeline import Pipeline

COLORS = {'female': (255, 0, 0), 'male': (0, 0, 255), 'both': (255, 0, 255)}  # BGR
WING_COLOR = (255, 255, 0)
WINDOW = 'Fruit fly analysis'


def arrow(img, start, length, angle, color):
    """Arrow from start at angle (radians, counter-clockwise, y up)."""
    tip = bound_point((start[0] + length * np.cos(angle), start[1] - length * np.sin(angle)), img)
    cv2.arrowedLine(img, start, tip, color, 5, tipLength=0.3)


def draw(out, results):
    for label, r in results.items():
        cv2.drawContours(out, [r['contour']], -1, COLORS[label], 3)
        center = bound_point(r['center'], out)
        cv2.circle(out, center, 5, COLORS[label], -1)
        if label == 'both':
            continue
        length = 0.3 * r['axes'][0]
        arrow(out, center, length, r['angle'], COLORS[label])
        if 'wing_right' in r:
            # wings point backwards from the body, rotated right/left by their angle
            arrow(out, center, length, r['angle'] + np.radians(r['wing_right']) - np.pi, WING_COLOR)
            arrow(out, center, length, r['angle'] - np.radians(r['wing_left']) - np.pi, WING_COLOR)
            cv2.putText(out, 'R {:5.1f}  L {:5.1f} deg'.format(r['wing_right'], r['wing_left']),
                        (center[0] + 40, center[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WING_COLOR, 3)


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-i', '--input', default='test4.mp4', help='video in the videos directory')
    parser.add_argument('--no-display', action='store_true', help='do not open a window')
    parser.add_argument('--write-video', action='store_true', help='save the overlay to results/demo_<input>')
    args = parser.parse_args()

    cap = cv2.VideoCapture(os.path.join(VIDEOS_DIR, args.input))
    if not cap.isOpened():
        raise FileNotFoundError(os.path.join(VIDEOS_DIR, args.input))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    writer = None
    if args.write_video:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, 'demo_' + os.path.splitext(args.input)[0] + '.mp4')
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, size)
    if not args.no_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, size[0] // 2, size[1] // 2)

    pipeline = Pipeline()
    frames, busy = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = perf_counter()
        results = pipeline.process(frame[:, :, 0])
        busy += perf_counter() - t0
        frames += 1

        draw(frame, results)
        cv2.putText(frame, 'pipeline: {:.0f} FPS'.format(frames / busy), (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 160, 0), 3)
        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(max(1, int(1e3 / fps_src - 1e3 * (perf_counter() - t0))))
            if key == ord(' '):
                key = cv2.waitKey(0)
            if key == ord('q') or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print('wrote', path)
    cv2.destroyAllWindows()
    print('{} frames, pipeline throughput {:.1f} FPS (excluding I/O and drawing)'.format(frames, frames / busy))


if __name__ == '__main__':
    main()
