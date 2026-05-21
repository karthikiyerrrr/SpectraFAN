"""Profiling harness (torch.profiler + optional Nsight wrappers).

Locked measurement protocol:
- warm up, then median + IQR over >= 100 runs at fixed batch size,
- torch.cuda.synchronize() around timing,
- per-op breakdown of one forward pass (fft/ifft vs. depthwise vs. 1x1 vs. other),
- kernel-launch count per FAM,
- memory-bandwidth utilization where available.

All numbers reported as deltas against the reproduced baseline, averaged over >= 3 seeds.
"""
