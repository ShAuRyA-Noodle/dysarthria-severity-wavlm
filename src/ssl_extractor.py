"""SSL feature extractor — freezes WavLM-Large and extracts per-layer embeddings."""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoModel
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────
MODELS_CONFIG = {
    "wavlm_large": {
        "hf_name": "microsoft/wavlm-large",
        "hidden_dim": 1024,
        "n_layers": 24,
        "feature_layer": "hidden_states",
    },
    "hubert_large": {
        "hf_name": "facebook/hubert-large-ll60k",
        "hidden_dim": 1024,
        "n_layers": 24,
        "feature_layer": "hidden_states",
    },
}

TARGET_LAYERS = [1, 3, 6, 9, 12, 15, 18, 21, 23]
# 9 representative layers across the 24-layer transformer

RANDOM_SEED = 42

logger = logging.getLogger(__name__)


class SSLExtractor:
    """
    Extracts multi-layer frame-level embeddings from frozen SSL models.
    Model is ALWAYS in eval mode with no_grad. Never fine-tuned.
    """

    def __init__(
        self,
        model_name: str = "wavlm_large",
        device: str = "auto",
        use_fp16: bool = True,
    ) -> None:
        """
        Load model from HuggingFace, freeze ALL parameters.
        Set model to eval() mode permanently.
        Log model size and VRAM usage after loading.

        Args:
            model_name: Key into MODELS_CONFIG.
            device: 'cuda', 'cpu', or 'auto' (auto-detect; default).
            use_fp16: Whether to use half-precision (CUDA only). Ignored on CPU.
        """
        if model_name not in MODELS_CONFIG:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODELS_CONFIG.keys())}"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.config = MODELS_CONFIG[model_name]
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"

        logger.info(f"Loading {self.config['hf_name']}...")

        # Load model — output_hidden_states is required
        self.model = AutoModel.from_pretrained(
            self.config["hf_name"],
            output_hidden_states=True,
        )

        # Freeze ALL parameters — NON-NEGOTIABLE
        for param in self.model.parameters():
            param.requires_grad = False

        # Eval mode permanently
        self.model.eval()

        # Move to device, optionally half-precision
        if self.use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(self.device)

        # Log model info
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {n_params / 1e6:.1f}M parameters")
        logger.info(f"Device: {self.device}, FP16: {self.use_fp16}")

        if self.device == "cuda":
            vram = self.get_vram_usage()
            logger.info(
                f"VRAM after loading: "
                f"{vram['allocated_gb']:.2f} GB allocated, "
                f"{vram['free_gb']:.2f} GB free"
            )

    @torch.no_grad()
    def extract_all_layers(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        target_layers: Optional[list[int]] = None,
    ) -> dict[int, np.ndarray]:
        """
        Extract frame-level embeddings from all target layers.

        Steps:
            1. Convert numpy to tensor, move to device
            2. Normalize: subtract mean, divide by std
            3. Forward pass with output_hidden_states=True
            4. Extract specified layers from hidden_states tuple
            5. Convert to numpy, move to CPU
            6. Return: {layer_idx: array of shape (T_frames, 1024)}

        T_frames ≈ audio_length_seconds * 50 (WavLM outputs ~50 fps)

        Args:
            waveform: 1-D numpy array of audio samples at sample_rate.
            sample_rate: Expected 16000.
            target_layers: Layer indices to extract. Defaults to TARGET_LAYERS.

        Returns:
            Dict mapping layer index → (T_frames, hidden_dim) numpy array.
        """
        if target_layers is None:
            target_layers = TARGET_LAYERS

        # 1. Convert to tensor
        if waveform.ndim == 1:
            wav_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        else:
            wav_tensor = torch.from_numpy(waveform)

        wav_tensor = wav_tensor.float()

        # 2. Normalize: subtract mean, divide by std
        mean = wav_tensor.mean()
        std = wav_tensor.std()
        if std > 0:
            wav_tensor = (wav_tensor - mean) / std

        # Move to device
        if self.use_fp16:
            wav_tensor = wav_tensor.half()
        wav_tensor = wav_tensor.to(self.device)

        # 3. Forward pass
        outputs = self.model(wav_tensor, output_hidden_states=True)

        # 4. Extract specified layers from hidden_states
        # hidden_states is a tuple of (n_layers + 1) tensors
        # Index 0 = CNN feature extractor output
        # Index 1..24 = transformer layer 1..24
        hidden_states = outputs.hidden_states

        result = {}
        for layer_idx in target_layers:
            if layer_idx < 0 or layer_idx >= len(hidden_states):
                logger.warning(
                    f"Layer {layer_idx} out of range "
                    f"(0..{len(hidden_states) - 1}), skipping"
                )
                continue
            # 5. Convert to numpy, move to CPU
            layer_emb = hidden_states[layer_idx].squeeze(0)  # (T_frames, 1024)
            result[layer_idx] = layer_emb.float().cpu().numpy()

        return result

    @torch.no_grad()
    def extract_utterance_embedding(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        layer: int = 9,
    ) -> np.ndarray:
        """
        Mean-pool frame embeddings → single utterance vector (1024,).
        Used for naive SSL baseline.

        Args:
            waveform: 1-D numpy array of audio samples.
            sample_rate: Expected 16000.
            layer: Which transformer layer to use.

        Returns:
            1-D numpy array of shape (1024,).
        """
        layer_embeddings = self.extract_all_layers(
            waveform, sample_rate, target_layers=[layer]
        )
        frame_embeddings = layer_embeddings[layer]  # (T_frames, 1024)
        return frame_embeddings.mean(axis=0)  # (1024,)

    def get_vram_usage(self) -> dict[str, float]:
        """
        Return current GPU VRAM usage.

        Returns:
            Dict with keys: allocated_gb, reserved_gb, free_gb.
        """
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - allocated

        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "free_gb": round(free, 3),
        }

    def get_model_info(self) -> dict:
        """Return model metadata for logging.

        Returns:
            Dict with model name, parameter count, device, fp16 status.
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "hf_name": self.config["hf_name"],
            "n_parameters": n_params,
            "n_parameters_M": round(n_params / 1e6, 1),
            "hidden_dim": self.config["hidden_dim"],
            "n_layers": self.config["n_layers"],
            "device": str(self.device),
            "use_fp16": self.use_fp16,
        }


# ── Severity & Gender Mappings (hardcoded, never change) ──────────────

TORGO_SEVERITY = {
    "F01": 2, "F03": 1, "F04": 2,
    "M01": 2, "M02": 2, "M03": 1, "M04": 2, "M05": 1,
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
}


def _load_extraction_log(log_path: Path) -> dict:
    """Load or create extraction log for resumability.

    Args:
        log_path: Path to extraction_log.json.

    Returns:
        Dict with keys: completed, failed, start_time, last_update.
    """
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed": [],
        "failed": [],
        "start_time": time.time(),
        "last_update": time.time(),
        "model_name": "",
        "target_layers": [],
    }


def _save_extraction_log(log: dict, log_path: Path) -> None:
    """Save extraction log atomically (write tmp then rename).

    Args:
        log: The log dict to save.
        log_path: Path to extraction_log.json.
    """
    log["last_update"] = time.time()
    tmp_path = log_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    tmp_path.replace(log_path)


def extract_dataset_embeddings(
    metadata_df: pd.DataFrame,
    model_name: str = "wavlm_large",
    output_dir: str = "data/embeddings/wavlm_large/",
    target_layers: Optional[list[int]] = None,
    batch_size: int = 1,
    resume: bool = True,
    project_root: Optional[Path] = None,
) -> str:
    """
    Extract embeddings for ALL utterances. MUST BE RESUMABLE.

    For each utterance:
    - Save as: {output_dir}/{speaker_id}/{utterance_id}.npy
    - Shape: (n_layers, T_frames, 1024) where n_layers=len(target_layers)
    - Also save layer indices as metadata: {output_dir}/layer_indices.json

    Resumability:
    - Maintain extraction_log.json tracking completed utterance_ids
    - On start: load log, skip already completed
    - On crash: log survives, resume from where it stopped

    Progress reporting every 50 utterances:
    - Utterances done/total
    - Estimated time remaining
    - Current VRAM usage
    - Any failures

    Args:
        metadata_df: DataFrame with utterance_id, speaker_id, filepath_processed.
        model_name: Key into MODELS_CONFIG.
        output_dir: Where to save .npy files.
        target_layers: Layer indices to extract.
        batch_size: Currently always 1 (per-utterance).
        resume: Whether to skip already-extracted utterances.
        project_root: Project root directory for resolving paths.

    Returns:
        Path to extraction_log.json.
    """
    if target_layers is None:
        target_layers = TARGET_LAYERS
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    log_path = output_path / "extraction_log.json"
    log = _load_extraction_log(log_path)

    # Save layer indices metadata
    layer_indices_path = output_path / "layer_indices.json"
    with open(layer_indices_path, "w", encoding="utf-8") as f:
        json.dump({"target_layers": target_layers}, f, indent=2)

    # Determine what to extract
    all_utt_ids = metadata_df["utterance_id"].tolist()
    completed_set = set(log["completed"])

    if resume:
        remaining = [uid for uid in all_utt_ids if uid not in completed_set]
        logger.info(
            f"Resuming: {len(completed_set)} done, "
            f"{len(remaining)} remaining, "
            f"{len(log['failed'])} previously failed"
        )
    else:
        remaining = all_utt_ids
        log = {
            "completed": [],
            "failed": [],
            "start_time": time.time(),
            "last_update": time.time(),
            "model_name": model_name,
            "target_layers": target_layers,
        }

    log["model_name"] = model_name
    log["target_layers"] = target_layers
    log["total"] = len(all_utt_ids)

    if not remaining:
        logger.info("All utterances already extracted. Nothing to do.")
        _save_extraction_log(log, log_path)
        return str(log_path)

    # Initialize extractor.
    # cudnn.benchmark intentionally NOT enabled — anti-deterministic.
    # Use src.seed_utils.set_global_seed() at the entry script for repro.
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    extractor = SSLExtractor(
        model_name=model_name, device="auto", use_fp16=True
    )

    # Build lookup: utterance_id → row
    meta_lookup = metadata_df.set_index("utterance_id")

    # Track timing
    start_time = time.time()
    batch_start = time.time()
    n_done_this_session = 0
    current_speaker = None

    print(f"\n{'='*60}")
    print(f"  SSL Extraction: {model_name}")
    print(f"  Remaining: {len(remaining)} / {len(all_utt_ids)} utterances")
    print(f"  Target layers: {target_layers}")
    print(f"{'='*60}\n")

    for i, utt_id in enumerate(tqdm(remaining, desc="Extracting")):
        try:
            row = meta_lookup.loc[utt_id]
            speaker_id = row["speaker_id"]

            # Clear VRAM cache between speakers
            if speaker_id != current_speaker:
                if current_speaker is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_speaker = speaker_id

            # Resolve audio path
            audio_path = project_root / row["filepath_processed"]
            if not audio_path.exists():
                logger.warning(f"Audio not found: {audio_path}")
                log["failed"].append(
                    {"utterance_id": utt_id, "reason": "file_not_found"}
                )
                _save_extraction_log(log, log_path)
                continue

            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            waveform = waveform.squeeze(0).numpy()  # (T_samples,)

            # Extract embeddings from all target layers
            layer_embeddings = extractor.extract_all_layers(
                waveform, sample_rate=sr, target_layers=target_layers
            )

            # Stack into (n_layers, T_frames, 1024)
            # All layers should have the same T_frames
            stacked = np.stack(
                [layer_embeddings[l] for l in target_layers], axis=0
            )

            # Save
            speaker_dir = output_path / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            npy_path = speaker_dir / f"{utt_id}.npy"
            np.save(str(npy_path), stacked)

            # Update log
            log["completed"].append(utt_id)
            n_done_this_session += 1

            # Progress reporting every 50 utterances
            if n_done_this_session % 50 == 0:
                elapsed = time.time() - start_time
                rate = n_done_this_session / elapsed
                eta = (len(remaining) - (i + 1)) / rate if rate > 0 else 0
                vram = extractor.get_vram_usage()
                print(
                    f"\n  [{n_done_this_session + len(completed_set)}"
                    f"/{len(all_utt_ids)}] "
                    f"Rate: {rate:.1f} utt/s | "
                    f"ETA: {eta / 60:.1f} min | "
                    f"VRAM: {vram['allocated_gb']:.2f} GB | "
                    f"Failed: {len(log['failed'])}"
                )
                _save_extraction_log(log, log_path)

        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA OOM on {utt_id} — clearing cache and skipping")
            torch.cuda.empty_cache()
            log["failed"].append(
                {"utterance_id": utt_id, "reason": "cuda_oom"}
            )
            _save_extraction_log(log, log_path)
            continue

        except Exception as e:
            logger.error(f"Failed on {utt_id}: {e}")
            log["failed"].append(
                {"utterance_id": utt_id, "reason": str(e)[:200]}
            )
            _save_extraction_log(log, log_path)
            continue

    # Final save
    log["end_time"] = time.time()
    log["total_time_sec"] = log["end_time"] - log.get("start_time", log["end_time"])
    _save_extraction_log(log, log_path)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Extraction Complete")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Completed: {len(log['completed'])} / {len(all_utt_ids)}")
    print(f"  Failed: {len(log['failed'])}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n")

    return str(log_path)


def build_ssl_index(
    metadata_df: pd.DataFrame,
    embedding_dirs: Optional[dict[str, str]] = None,
    output_path: str = "data/embeddings/ssl_index.json",
    project_root: Optional[Path] = None,
) -> dict:
    """
    Build master index mapping utterance_id to all embedding file paths.

    Schema:
    {
      "TORGO_MC01_uw1a_S1_001": {
        "wavlm_large": "data/embeddings/wavlm_large/MC01/TORGO_MC01_uw1a_S1_001.npy",
        "hubert_large": null,
        "shape": [9, 47, 1024],
        "layer_indices": [1, 3, 6, 9, 12, 15, 18, 21, 23],
        "speaker": "MC01",
        "word": "uw1a",
        "severity": 2,
        "is_control": false,
        "duration_sec": 0.94
      }
    }

    Args:
        metadata_df: DataFrame with utterance metadata.
        embedding_dirs: Dict mapping model_name → directory path.
        output_path: Where to save the index JSON.
        project_root: Project root for resolving paths.

    Returns:
        The index dict.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    if embedding_dirs is None:
        embedding_dirs = {
            "wavlm_large": "data/embeddings/wavlm_large/",
            "hubert_large": "data/embeddings/hubert_large/",
        }

    index = {}

    for _, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Building SSL index"
    ):
        utt_id = row["utterance_id"]
        speaker_id = row["speaker_id"]
        entry = {
            "speaker": speaker_id,
            "word": row.get("word", ""),
            "severity": int(TORGO_SEVERITY.get(speaker_id, -1)),
            "is_control": bool(TORGO_SEVERITY.get(speaker_id, -1) == 0),
            "duration_sec": float(row.get("duration_sec", 0)),
        }

        for model_name, emb_dir in embedding_dirs.items():
            npy_path = project_root / emb_dir / speaker_id / f"{utt_id}.npy"
            if npy_path.exists():
                # Use relative path in index for portability
                rel_path = str(npy_path.relative_to(project_root)).replace(
                    "\\", "/"
                )
                entry[model_name] = rel_path

                # Read shape from first match
                if "shape" not in entry:
                    emb = np.load(str(npy_path))
                    entry["shape"] = list(emb.shape)

                    # Load layer indices if available
                    layer_idx_path = project_root / emb_dir / "layer_indices.json"
                    if layer_idx_path.exists():
                        with open(layer_idx_path, "r") as f:
                            entry["layer_indices"] = json.load(f).get(
                                "target_layers", TARGET_LAYERS
                            )
                    else:
                        entry["layer_indices"] = TARGET_LAYERS
            else:
                entry[model_name] = None

        index[utt_id] = entry

    # Save
    out_file = project_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    n_with_wavlm = sum(1 for v in index.values() if v.get("wavlm_large"))
    n_with_hubert = sum(1 for v in index.values() if v.get("hubert_large"))
    print(f"SSL Index built: {len(index)} utterances")
    print(f"  wavlm_large: {n_with_wavlm} embeddings")
    print(f"  hubert_large: {n_with_hubert} embeddings")
    print(f"  Saved to: {out_file}")

    return index
