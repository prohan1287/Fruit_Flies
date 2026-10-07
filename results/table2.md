# Table 2: timing breakdown

Video: test4.mp4 (632 frames, 0 not analysed), median of 5 runs
Our hardware: 13th Gen Intel(R) Core(TM) i7-13700HX, 16 GB RAM, 24 logical cores, no GPU
Paper hardware: 2.8 GHz Intel Core i7, 16 GB RAM, no GPU

| Step | Runtime (ours) | % total (ours) | Runtime (paper) | % total (paper) |
|---|---|---|---|---|
| I/O | 7.9 ms | 29.8% | 3.9 ms | 32.5% |
| FlyCount | 8.0 ms | 30.1% | 2.2 ms | 18.3% |
| ♂♀ vs ♀♂ | 1.7 ms | 6.3% | 0.7 ms | 5.8% |
| Orientation | 3.8 ms | 14.4% | 1.9 ms | 15.8% |
| WingAngle | 5.2 ms | 19.5% | 3.3 ms | 27.5% |

Throughput (ours): 37.8 FPS (26.4 ms/frame); individual runs: 27.4, 21.7, 37.8, 71.8, 71.5 FPS
Throughput (paper): 84.0 FPS (11.9 ms/frame)
Speed-up over the 30 FPS source video: ours 1.3x, paper 2.8x (paper text says 2.4x)
