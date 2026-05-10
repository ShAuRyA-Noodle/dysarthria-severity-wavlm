"""Per-word reference trajectory cache for streaming ATD (Track A)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


class ReferenceCache:
    """Lazy-loads the per-word PCA-64 control reference trajectories built
    by `src.reference_builder.build_word_reference`.

    Holds at most `lru_capacity` references in memory; falls back to disk
    for cold words. PCA model is also cached so utterance projection
    matches reference projection (same coordinate space).
    """

    def __init__(
        self,
        manifest_path: Path,
        project_root: Path,
        layer: int = 9,
        lru_capacity: int = 64,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.layer = int(layer)
        self.lru_capacity = int(lru_capacity)
        with open(self.manifest_path, encoding="utf-8") as f:
            self._manifest = json.load(f)
        self._refs: dict[str, dict] = {}
        self._lru: list[str] = []

    def has(self, word: str) -> bool:
        key = f"{word}_{self.layer}"
        return key in self._manifest.get("word_layer_ref", {})

    def get(self, word: str) -> Optional[dict]:
        """Return dict with keys 'reference' (np.ndarray) and 'pca_model'."""
        key = f"{word}_{self.layer}"
        if key in self._refs:
            self._touch(key)
            return self._refs[key]
        info = self._manifest.get("word_layer_ref", {}).get(key)
        if info is None:
            return None
        pkl_path = self.project_root / info["path"]
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._refs[key] = data
        self._touch(key)
        return data

    def project(self, word: str, frames: np.ndarray) -> Optional[np.ndarray]:
        """Project (T, 1024) WavLM frames to (T, 64) PCA reference space."""
        ref = self.get(word)
        if ref is None:
            return None
        pca = ref.get("pca_model")
        if pca is None:
            return None
        return pca.transform(frames)

    def _touch(self, key: str) -> None:
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)
        while len(self._lru) > self.lru_capacity:
            evict = self._lru.pop(0)
            self._refs.pop(evict, None)
