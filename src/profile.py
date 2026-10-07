"""Reproduce Table 2 (per-stage runtime) and overall throughput, next to the paper's numbers.

Processes a test video without display, several times, and reports the median.
Writes results/table2.md.

    python src/profile.py -i test4.mp4 -n 5
"""

import os
import platform
import subprocess
from argparse import ArgumentParser
from time import perf_counter

import cv2
import numpy as np

from data import RESULTS_DIR, VIDEOS_DIR
from pipeline import STAGE_NAMES, Pipeline, Profiler

PAPER_MS = {'I/O': 3.9, 'FlyCount': 2.2, '♂♀ vs ♀♂': 0.7, 'Orientation': 1.9, 'WingAngle': 3.3}
PAPER_FPS = 84.0
PAPER_HW = '2.8 GHz Intel Core i7, 16 GB RAM, no GPU'
SOURCE_FPS = 30  # poster


def hardware():
    cpu = platform.processor()
    ram = ''
    if platform.system() == 'Windows':
        try:
            out = subprocess.run(['powershell', '-NoProfile', '-Command',
                                  '(Get-CimInstance Win32_Processor).Name;'
                                  '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)'],
                                 capture_output=True, text=True, timeout=30).stdout.split('\n')
            cpu, ram = out[0].strip(), out[1].strip() + ' GB RAM'
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return ', '.join(x for x in [cpu, ram, '{} logical cores'.format(os.cpu_count()), 'no GPU'] if x)


def run_once(pipeline, path):
    """One pass over the video. Returns (frames, frames not analysed, seconds, Profiler)."""
    cap = cv2.VideoCapture(path)
    prof = Profiler()
    frames = skipped = 0
    start = perf_counter()
    while True:
        prof.tick('I/O')
        ok, frame = cap.read()
        if ok:
            img = frame[:, :, 0]  # grayscale video: all three channels are equal (as in author code)
        prof.tock('I/O')
        if not ok:
            break
        frames += 1
        if not pipeline.process(img, prof):
            skipped += 1
    elapsed = perf_counter() - start
    cap.release()
    return frames, skipped, elapsed, prof


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-i', '--input', default='test4.mp4', help='video in the videos directory')
    parser.add_argument('-n', '--runs', type=int, default=5, help='passes over the video; the median is reported')
    args = parser.parse_args()

    pipeline = Pipeline()
    runs = [run_once(pipeline, os.path.join(VIDEOS_DIR, args.input)) for _ in range(args.runs)]
    frames, skipped = runs[0][0], runs[0][1]
    fps_all = [r[0] / r[2] for r in runs]

    # median over runs of each step's per-frame mean (later stages only run on frames that reach them)
    rows = [(name, float(np.median([r[3].mean_ms(name) for r in runs]))) for name in STAGE_NAMES]
    total = sum(ms for _, ms in rows)
    paper_total = sum(PAPER_MS.values())
    out = ['# Table 2: timing breakdown', '',
           'Video: {} ({} frames, {} not analysed), median of {} runs'.format(
               args.input, frames, skipped, args.runs),
           'Our hardware: {}'.format(hardware()), 'Paper hardware: {}'.format(PAPER_HW), '',
           '| Step | Runtime (ours) | % total (ours) | Runtime (paper) | % total (paper) |',
           '|---|---|---|---|---|']
    for name, ms in rows:
        out.append('| {} | {:.1f} ms | {:.1f}% | {} ms | {:.1f}% |'.format(
            name, ms, 100 * ms / total, PAPER_MS[name], 100 * PAPER_MS[name] / paper_total))
    fps = float(np.median(fps_all))
    out += ['',
            'Throughput (ours): {:.1f} FPS ({:.1f} ms/frame); individual runs: {} FPS'.format(
                fps, 1e3 / fps, ', '.join('{:.1f}'.format(f) for f in fps_all)),
            'Throughput (paper): {:.1f} FPS (11.9 ms/frame)'.format(PAPER_FPS),
            'Speed-up over the {} FPS source video: ours {:.1f}x, paper {:.1f}x (paper text says 2.4x)'.format(
                SOURCE_FPS, fps / SOURCE_FPS, PAPER_FPS / SOURCE_FPS)]

    text = '\n'.join(out)
    print(text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'table2.md'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
