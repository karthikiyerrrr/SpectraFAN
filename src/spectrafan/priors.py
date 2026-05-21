"""Phase 2 — crystallographic frequency priors.

Mechanisms (cheapest -> most invasive):
- prior-based initialization of FAM filters from expected lattice reciprocal vectors,
- soft spectral regularization that decays over training,
- synthetic-spectral pretraining on procedurally generated periodic patterns,
- (stretch) hard spectral truncation as a prior.
"""
