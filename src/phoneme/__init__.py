"""Phoneme-level ATD localization (Track B).

Pipeline: MFA forced alignment → DTW path slicing → per-phoneme deviation
+ z-scoring against control-only distributions → heatmap viz.
"""
