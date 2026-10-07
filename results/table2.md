# Table 2: timing breakdown

Video: test4.mp4 (632 frames, 0 not analysed), median of 5 runs
Our hardware: 13th Gen Intel(R) Core(TM) i7-13700HX, 16 GB RAM, 24 logical cores, no GPU
Paper hardware: 2.8 GHz Intel Core i7, 16 GB RAM, no GPU

| Step | Runtime (ours) | % total (ours) | Runtime (paper) | % total (paper) |
|---|---|---|---|---|
| I/O | 4.3 ms | 31.2% | 3.9 ms | 32.5% |
| FlyCount | 4.5 ms | 32.3% | 2.2 ms | 18.3% |
| ♂♀ vs ♀♂ | 0.8 ms | 5.9% | 0.7 ms | 5.8% |
| Orientation | 1.7 ms | 12.5% | 1.9 ms | 15.8% |
| WingAngle | 2.5 ms | 18.1% | 3.3 ms | 27.5% |

Throughput (ours): 72.0 FPS (13.9 ms/frame); individual runs: 35.0, 68.6, 72.5, 72.2, 72.0 FPS
Throughput (paper): 84.0 FPS (11.9 ms/frame)
Speed-up over the 30 FPS source video: ours 2.4x, paper 2.8x (paper text says 2.4x)
