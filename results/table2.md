# Table 2: timing breakdown

Video: test4.mp4 (632 frames, 0 not analysed), median of 5 runs
Our hardware: 13th Gen Intel(R) Core(TM) i7-13700HX, 16 GB RAM, 24 logical cores, no GPU
Paper hardware: 2.8 GHz Intel Core i7, 16 GB RAM, no GPU

| Step | Runtime (ours) | % total (ours) | Runtime (paper) | % total (paper) |
|---|---|---|---|---|
| I/O | 4.3 ms | 31.1% | 3.9 ms | 32.5% |
| FlyCount | 4.5 ms | 32.5% | 2.2 ms | 18.3% |
| ♂♀ vs ♀♂ | 0.8 ms | 6.0% | 0.7 ms | 5.8% |
| Orientation | 1.7 ms | 12.4% | 1.9 ms | 15.8% |
| WingAngle | 2.5 ms | 18.1% | 3.3 ms | 27.5% |

Throughput (ours): 72.5 FPS (13.8 ms/frame); individual runs: 31.5, 72.1, 73.7, 74.0, 72.5 FPS
Throughput (paper): 84.0 FPS (11.9 ms/frame)
Speed-up over the 30 FPS source video: ours 2.4x, paper 2.8x (paper text says 2.4x)
