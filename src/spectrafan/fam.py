"""Frequency Attention Module variants.

Planned implementations:
- FAMComplex: faithful reproduction of the published FAM (fft2 -> split real/imag
  -> per-part depthwise convs -> ifft2 -> merge -> 1x1 conv).
- FAMReal:    rfft2 / irfft2 over the half-spectrum (Phase 1, Step A).
- FAMDCT:     real-valued DCT-domain variant, optionally IFFT-free (Phase 1, Step C).
"""
