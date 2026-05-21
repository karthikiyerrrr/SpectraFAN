"""torch.compile and CUDA-graph wrappers (Phase 1, Step B).

Static-shape inference path is ideal for CUDA graph capture. Where torch.compile
will not fuse across the FFT boundary, fall back to a manually fused elementwise
chain (split / bn / relu / combine).
"""
