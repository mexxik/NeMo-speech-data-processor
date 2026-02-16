# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Batched Whisper inference processor for NeMo Speech Data Processor.

Drop-in replacement for FasterWhisperInference that uses CTranslate2's
batch inference API for significantly higher throughput (8-16x).

Instead of processing one audio file at a time, this processor batches
multiple files together and feeds them through CTranslate2's Whisper
model simultaneously.

Supports three modes:
  - Language detection only (language_detection_only=True)
  - Full transcription with timestamps (default)
  - Offset-sliced segment transcription (slice_by_offset=True)

Example YAML config:
    - _target_: sdp.processors.BatchedWhisperInference
      model_size_or_path: large-v3
      batch_size: 16
      device: cuda
      compute_type: float16
      output_manifest_file: ${workspace_dir}/manifest_06.json
      output_dir: ${workspace_dir}/step_06
      inference:
        language: en
      save_timestamps_separately: False
"""

import json
import os
import traceback
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor


# Whisper special token IDs (multilingual, 99 languages)
SOT = 50258           # <|startoftranscript|>
EOT = 50257           # <|endoftext|>
TRANSLATE = 50358     # <|translate|>
TRANSCRIBE = 50359    # <|transcribe|>
NO_TIMESTAMPS = 50363 # <|notimestamps|>
TIMESTAMP_BEGIN = 50364  # <|0.00|>
NO_SPEECH = 50362     # <|nospeech|>

# Language code to token ID mapping (offset from SOT+1)
_LANG_CODES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
]
LANG_TO_TOKEN = {lang: SOT + 1 + i for i, lang in enumerate(_LANG_CODES)}
TOKEN_TO_LANG = {v: k for k, v in LANG_TO_TOKEN.items()}


def _gpu_worker(processor, gpu_id, batch_queue, result_queue, total_entries=0):
    """Worker with GPU-based mel feature extraction.

    CPU threads load raw audio only (I/O-bound, releases GIL).
    Mel spectrogram computation runs on GPU via PyTorch STFT — replaces the
    CPU feature extraction bottleneck (~4-12s) with ~50ms of GPU work.
    """
    try:
        import queue as _queue_mod
        import time
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        import ctranslate2
        import torch

        processor._load_model(gpu_id)

        # ── GPU mel feature extraction setup ──────────────────────────
        device = torch.device(f"cuda:{gpu_id}")
        n_fft = 400
        hop_length = 160
        N_FRAMES = 3000  # 30s at 100 fps

        # Mel filterbank from faster_whisper (shape: 80 × n_freq)
        mel_filters_np = processor.feature_extractor.mel_filters
        n_freq = mel_filters_np.shape[1]  # 200 for faster_whisper
        mel_filters_gpu = torch.from_numpy(mel_filters_np).float().to(device)

        # Periodic Hann window — matches np.hanning(n_fft+1)[:-1]
        hann_window = torch.hann_window(n_fft, periodic=True, device=device)

        def _extract_features_gpu(audio_arrays):
            """Batched mel spectrogram on GPU. Replicates faster_whisper's
            FeatureExtractor exactly: STFT → power spectrum → mel → log norm.
            """
            max_len = max(a.shape[0] for a in audio_arrays)
            B = len(audio_arrays)

            # Pad all audio to same length, build batch tensor
            batch_np = np.zeros((B, max_len), dtype=np.float32)
            for i, audio in enumerate(audio_arrays):
                batch_np[i, : len(audio)] = audio
            batch_audio = torch.from_numpy(batch_np).to(device)

            # Batched STFT  (center=False matches faster_whisper framing)
            stft = torch.stft(
                batch_audio,
                n_fft=n_fft,
                hop_length=hop_length,
                window=hann_window,
                center=False,
                return_complex=True,
            )
            # stft: (B, n_fft//2+1, T) = (B, 201, T)
            # Keep first n_freq bins to match faster_whisper's stft[:, :-1]
            magnitudes = stft[:, :n_freq, :].abs().pow(2)

            # Mel filterbank: (80, n_freq) @ (B, n_freq, T) → (B, 80, T)
            mel_spec = torch.matmul(mel_filters_gpu, magnitudes)

            # Log scale + dynamic range compression (per-sample)
            log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
            log_max = log_spec.flatten(1).max(dim=1).values[:, None, None]
            log_spec = torch.maximum(log_spec, log_max - 8.0)
            log_spec = (log_spec + 4.0) / 4.0

            # Pad / truncate to N_FRAMES
            T = log_spec.shape[2]
            if T < N_FRAMES:
                log_spec = torch.nn.functional.pad(log_spec, (0, N_FRAMES - T))
            elif T > N_FRAMES:
                log_spec = log_spec[:, :, :N_FRAMES]

            return log_spec.cpu().numpy()  # (B, 80, 3000)

        # ── Audio loading (CPU threads, I/O-bound) ───────────────────
        import soundfile as sf

        def _load_audio_only(entry):
            """Load one audio file. Uses soundfile for 43x faster WAV reads
            vs PyAV. Falls back to processor._load_audio for slice_by_offset."""
            try:
                if processor.slice_by_offset:
                    audio = processor._load_audio(entry)
                    return entry, audio
                filepath = entry[processor.audio_filepath_field]
                audio, sr = sf.read(filepath, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return entry, audio
            except Exception:
                if processor.skip_corrupted:
                    return entry, None
                raise

        worker_pool = ThreadPoolExecutor(max_workers=12)

        def _prefetch_audio(batch_entries):
            """Load audio for a whole batch in parallel CPU threads."""
            t0 = time.monotonic()
            futs = [worker_pool.submit(_load_audio_only, e) for e in batch_entries]
            results = [f.result() for f in futs]

            valid_entries = []
            audio_list = []
            skipped = 0
            for entry, audio in results:
                if audio is not None:
                    valid_entries.append(entry)
                    audio_list.append(audio)
                else:
                    skipped += 1

            return valid_entries, audio_list, skipped, time.monotonic() - t0

        # ── Prefetch pipeline ─────────────────────────────────────────
        PREFETCH_DEPTH = 3
        prefetch_pool = ThreadPoolExecutor(max_workers=2)
        prefetch_deque = deque()

        done_feeding = False
        for _ in range(PREFETCH_DEPTH):
            item = batch_queue.get()
            if item is None:
                done_feeding = True
                break
            prefetch_deque.append(prefetch_pool.submit(_prefetch_audio, item))

        all_results = []
        total_skipped = 0
        batch_count = 0
        t_start = time.monotonic()
        cum_wait = 0.0
        cum_feat = 0.0
        cum_infer = 0.0

        while prefetch_deque:
            # Wait for next batch of raw audio
            t0 = time.monotonic()
            valid_entries, audio_list, skipped, t_audio = prefetch_deque.popleft().result()
            t_wait = time.monotonic() - t0
            cum_wait += t_wait

            total_skipped += skipped
            batch_count += 1

            t_feat = 0.0
            t_infer = 0.0
            if valid_entries:
                # GPU feature extraction (replaces CPU bottleneck)
                t0 = time.monotonic()
                features_np = _extract_features_gpu(audio_list)
                features = ctranslate2.StorageView.from_array(features_np)
                t_feat = time.monotonic() - t0
                cum_feat += t_feat

                # GPU inference
                t0 = time.monotonic()
                results = processor._run_inference(valid_entries, features)
                t_infer = time.monotonic() - t0
                cum_infer += t_infer
                all_results.extend(results)

            # Replenish prefetch
            if not done_feeding:
                while len(prefetch_deque) < PREFETCH_DEPTH:
                    try:
                        item = batch_queue.get_nowait()
                    except _queue_mod.Empty:
                        break
                    if item is None:
                        done_feeding = True
                        break
                    prefetch_deque.append(prefetch_pool.submit(_prefetch_audio, item))

                if not prefetch_deque and not done_feeding:
                    item = batch_queue.get()
                    if item is None:
                        done_feeding = True
                    else:
                        prefetch_deque.append(prefetch_pool.submit(_prefetch_audio, item))

            # Progress
            elapsed = time.monotonic() - t_start
            done_count = len(all_results) + total_skipped
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (total_entries - done_count) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            progress = f"{done_count}/{total_entries}" if total_entries else str(done_count)
            logger.info(
                f"GPU:{gpu_id} | batch {batch_count} | {progress} "
                f"| {rate:.0f} items/s | ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s "
                f"| wait={t_wait:.3f}s audio={t_audio:.3f}s "
                f"feat={t_feat:.3f}s infer={t_infer:.3f}s "
                f"| cum: wait={cum_wait:.1f}s feat={cum_feat:.1f}s infer={cum_infer:.1f}s"
            )

        elapsed = time.monotonic() - t_start
        logger.info(
            f"GPU:{gpu_id} finished: {len(all_results)} processed, "
            f"{total_skipped} skipped in {elapsed:.1f}s"
        )
        result_queue.put((gpu_id, all_results, total_skipped, None))
        prefetch_pool.shutdown(wait=False)
        worker_pool.shutdown(wait=False)
    except Exception as e:
        import traceback as _tb
        logger.error(f"GPU:{gpu_id} worker error: {e}\n{_tb.format_exc()}")
        result_queue.put((gpu_id, [], 0, e))


def _parse_timestamp_tokens(token_ids, tokenizer):
    """Parse Whisper output tokens into text and timestamp segments.

    Args:
        token_ids: Output token IDs from CTranslate2 generate().
        tokenizer: HuggingFace WhisperTokenizer for decoding text.

    Returns:
        Dict with "text" (full text) and "segments" (list of segment dicts).
    """
    segments = []
    current_text_tokens = []
    current_start = None
    segment_id = 0

    for token_id in token_ids:
        if token_id == EOT:
            break

        if token_id >= TIMESTAMP_BEGIN:
            timestamp = (token_id - TIMESTAMP_BEGIN) * 0.02
            if current_start is None:
                current_start = timestamp
            else:
                text = tokenizer.decode(current_text_tokens, skip_special_tokens=False).strip()
                if text:
                    segments.append({
                        "id": segment_id,
                        "start": round(current_start, 2),
                        "end": round(timestamp, 2),
                        "text": text,
                    })
                    segment_id += 1
                current_text_tokens = []
                current_start = timestamp
        elif token_id < SOT:
            # Regular text token (below special token range)
            current_text_tokens.append(token_id)

    # Handle trailing text
    if current_text_tokens:
        text = tokenizer.decode(current_text_tokens, skip_special_tokens=False).strip()
        if text:
            segments.append({
                "id": segment_id,
                "start": round(current_start or 0.0, 2),
                "end": None,
                "text": text,
            })

    full_text = " ".join(seg["text"] for seg in segments).strip()
    return {"text": full_text, "segments": segments}


class BatchedWhisperInference(BaseProcessor):
    """Batched Whisper transcription using CTranslate2.

    Reads a manifest of audio files, transcribes them in batches using
    CTranslate2's Whisper batch API, and writes results in a
    NeMo-compatible manifest.

    Args:
        input_manifest_file: Path to input manifest.
        output_manifest_file: Path to output manifest.
        model_size_or_path: Whisper model name or path (e.g. 'large-v3').
        batch_size: Number of audio files to process per batch.
        device: Device for inference ('cuda' or 'cpu').
        compute_type: CTranslate2 compute type ('float16', 'int8', 'float32').
        output_dir: Directory for intermediate output files.
        skip_corrupted_audios: Skip files that raise exceptions.
        save_timestamps_separately: Save timestamps to separate files.
        slice_by_offset: Slice audio using offset/duration fields.
        language_detection_only: Only detect language, don't transcribe.
        inference: Dict of inference parameters (language, beam_size, etc).
        in_memory_chunksize: Manifest entries to load per chunk.
        audio_filepath_field: Manifest field name for audio path.
    """

    def __init__(
        self,
        input_manifest_file: str,
        output_manifest_file: Optional[str] = None,
        model_size_or_path: str = "large-v3",
        batch_size: int = 16,
        device: str = "cuda",
        compute_type: str = "float16",
        num_devices: int = 1,
        output_dir: Optional[str] = None,
        skip_corrupted_audios: bool = False,
        save_timestamps_separately: bool = True,
        slice_by_offset: bool = False,
        language_detection_only: bool = False,
        inference: Optional[Dict] = None,
        in_memory_chunksize: int = 100000,
        audio_filepath_field: str = "audio_filepath",
    ):
        super().__init__(
            input_manifest_file=input_manifest_file,
            output_manifest_file=output_manifest_file,
        )

        if not self.output_manifest_file and not output_dir:
            raise ValueError("Either output_manifest_file or output_dir must be provided.")
        if not output_dir:
            output_dir = os.path.splitext(self.output_manifest_file)[0]
        if not self.output_manifest_file:
            self.output_manifest_file = os.path.join(output_dir, "predictions_all.json")

        self.model_size_or_path = model_size_or_path
        self.batch_size = batch_size
        self.device = device
        self.compute_type = compute_type
        self.num_devices = num_devices
        self.output_dir = output_dir
        self.skip_corrupted = skip_corrupted_audios
        self.save_timestamps_separately = save_timestamps_separately
        self.slice_by_offset = slice_by_offset
        self.language_detection_only = language_detection_only
        self.inference_cfg = inference or {}
        self.in_memory_chunksize = in_memory_chunksize
        self.audio_filepath_field = audio_filepath_field

        # Extract inference parameters
        self.language = self.inference_cfg.get("language", None)
        self.task = self.inference_cfg.get("task", "transcribe")
        self.beam_size = self.inference_cfg.get("beam_size", 5)
        self.best_of = self.inference_cfg.get("best_of", 5)
        self.patience = self.inference_cfg.get("patience", 1.0)
        self.length_penalty = self.inference_cfg.get("length_penalty", 1.0)
        self.repetition_penalty = self.inference_cfg.get("repetition_penalty", 1.0)
        self.no_repeat_ngram_size = self.inference_cfg.get("no_repeat_ngram_size", 0)
        self.temperature = self.inference_cfg.get("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.compression_ratio_threshold = self.inference_cfg.get("compression_ratio_threshold", 2.4)
        self.suppress_tokens = self.inference_cfg.get("suppress_tokens", [-1])

    def _load_model(self, device_index: int = 0):
        """Load FasterWhisper model and extract CTranslate2 components.

        Args:
            device_index: GPU index to load the model on (0-based).
        """
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio

        logger.info(
            f"Loading Whisper model: {self.model_size_or_path} ({self.compute_type}) "
            f"on {self.device}:{device_index}"
        )
        self.fw_model = WhisperModel(
            self.model_size_or_path,
            device=self.device,
            compute_type=self.compute_type,
            device_index=device_index,
        )
        self.ct2_model = self.fw_model.model  # ctranslate2.models.Whisper
        self.feature_extractor = self.fw_model.feature_extractor
        self.tokenizer = self.fw_model.hf_tokenizer
        self._decode_audio = decode_audio

        logger.info(
            f"Model loaded on {self.ct2_model.device}. Batch size: {self.batch_size}"
        )

    def _read_manifest(self):
        """Yield manifest entries from input file."""
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            for line in fin:
                yield json.loads(line)

    def _chunk_manifest(self):
        """Split manifest into memory-friendly chunks."""
        chunk = []
        for idx, entry in enumerate(self._read_manifest(), 1):
            chunk.append(entry)
            if idx % self.in_memory_chunksize == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _load_audio(self, entry: Dict) -> Optional[np.ndarray]:
        """Load audio for a manifest entry as float32 numpy array."""
        audio_filepath = entry[self.audio_filepath_field]
        try:
            if self.slice_by_offset:
                audio, sr = librosa.load(audio_filepath, sr=None)
                start = int(entry["offset"] * sr)
                end = int((entry["offset"] + entry["duration"]) * sr)
                audio = audio[start:end].astype(np.float32)
            else:
                audio = self._decode_audio(audio_filepath)
            return audio
        except Exception:
            if self.skip_corrupted:
                logger.warning(f"Failed to load audio: {audio_filepath}. Skipping.")
                return None
            traceback.print_exc()
            raise

    def _extract_features_batch(self, audio_list: List[np.ndarray]) -> np.ndarray:
        """Compute mel spectrograms for a batch of audio arrays.

        Pads all features to 3000 frames (30s Whisper window) for uniform batching.

        Returns:
            numpy array of shape (batch_size, n_mels, 3000).
        """
        N_FRAMES = 3000  # 30 seconds at 100 fps
        features = []
        for audio in audio_list:
            feat = self.feature_extractor(audio)  # (n_mels, T) where T varies
            n_mels, t = feat.shape
            if t < N_FRAMES:
                # Pad with zeros (silence) to 30s
                padded = np.zeros((n_mels, N_FRAMES), dtype=feat.dtype)
                padded[:, :t] = feat
                feat = padded
            elif t > N_FRAMES:
                # Truncate to 30s (long audio handled by chunking in transcription mode)
                feat = feat[:, :N_FRAMES]
            features.append(feat)
        import ctranslate2
        return ctranslate2.StorageView.from_array(np.stack(features))  # (batch, n_mels, 3000)

    def _build_prompts(self, batch_size: int, with_timestamps: bool) -> List[List[int]]:
        """Build decoder prompt token IDs for the batch.

        Args:
            batch_size: Number of items in the batch.
            with_timestamps: If True, enable timestamp token generation.

        Returns:
            List of prompt token ID lists, one per batch item.
        """
        task_token = TRANSCRIBE if self.task == "transcribe" else TRANSLATE

        if self.language:
            lang_token = LANG_TO_TOKEN.get(self.language)
            if lang_token is None:
                raise ValueError(f"Unknown language code: {self.language}")
            prompt = [SOT, lang_token, task_token]
        else:
            prompt = [SOT, task_token]

        if not with_timestamps:
            prompt.append(NO_TIMESTAMPS)

        return [prompt] * batch_size

    def _process_lid_batch(self, entries: List[Dict], features) -> List[Dict]:
        """Batch language detection using pre-computed features."""
        lang_results = self.ct2_model.detect_language(features)

        results = []
        for entry, lang_probs in zip(entries, lang_results):
            if lang_probs:
                top_lang, top_prob = lang_probs[0]
                entry["language"] = top_lang
                entry["language_probability"] = round(top_prob, 4)
            else:
                entry["language"] = "unknown"
                entry["language_probability"] = 0.0
            results.append(entry)

        return results

    def _process_transcription_batch(self, entries: List[Dict], features) -> List[Dict]:
        """Batch transcription with timestamps using pre-computed features."""
        prompts = self._build_prompts(len(entries), with_timestamps=True)

        gen_results = self.ct2_model.generate(
            features,
            prompts=prompts,
            beam_size=self.beam_size,
            patience=self.patience,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            suppress_tokens=self.suppress_tokens,
            suppress_blank=True,
            return_no_speech_prob=True,
        )

        results = []
        for entry, gen_result in zip(entries, gen_results):
            token_ids = gen_result.sequences_ids[0]  # Best hypothesis
            parsed = _parse_timestamp_tokens(token_ids, self.tokenizer)

            entry["pred_text"] = parsed["text"]
            entry["language"] = self.language or "unknown"
            entry["language_probability"] = 1.0

            if self.save_timestamps_separately:
                entry.update(
                    self._write_timestamps(entry[self.audio_filepath_field], parsed["segments"])
                )
            else:
                entry["segments"] = parsed["segments"]

            results.append(entry)

        return results

    def _process_offset_batch(self, entries: List[Dict], features) -> List[Dict]:
        """Batch transcription for offset-sliced segments using pre-computed features."""
        prompts = self._build_prompts(len(entries), with_timestamps=False)

        gen_results = self.ct2_model.generate(
            features,
            prompts=prompts,
            beam_size=self.beam_size,
            patience=self.patience,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            suppress_tokens=self.suppress_tokens,
            suppress_blank=True,
        )

        results = []
        for entry, gen_result in zip(entries, gen_results):
            token_ids = gen_result.sequences_ids[0]
            # Filter out any special tokens, keep only text
            text_tokens = [t for t in token_ids if t < SOT and t != EOT]
            pred_text = self.tokenizer.decode(text_tokens, skip_special_tokens=False).strip()

            entry["pred_text"] = pred_text
            entry["language"] = self.language or "unknown"
            entry["language_probability"] = 1.0
            results.append(entry)

        return results

    def _write_timestamps(self, audio_filepath: str, segments: List[Dict]) -> Dict:
        """Save timestamp segments to a separate JSON file."""
        filename = os.path.splitext(os.path.basename(audio_filepath))[0]
        segments_dir = os.path.join(self.output_dir, "segments")
        output_path = os.path.join(segments_dir, f"{filename}.json")

        with open(output_path, "w", encoding="utf8") as f:
            for segment in segments:
                f.write(json.dumps(segment, ensure_ascii=False) + "\n")

        return {"segments": output_path}

    def _prepare(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_timestamps_separately:
            os.makedirs(os.path.join(self.output_dir, "segments"), exist_ok=True)

    def _prefetch_batch(self, batch_entries: List[Dict]) -> tuple:
        """Load audio AND compute features for a batch (runs in background thread).

        Parallelizes audio loading with threads, then computes mel features.
        Returns (valid_entries, features_StorageView, num_skipped).
        """
        from concurrent.futures import ThreadPoolExecutor

        # Parallel audio loading (I/O-bound, threads work fine)
        num_workers = min(len(batch_entries), 8)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            audio_results = list(pool.map(self._load_audio, batch_entries))

        audio_list = []
        valid_entries = []
        num_skipped = 0
        for entry, audio in zip(batch_entries, audio_results):
            if audio is not None:
                audio_list.append(audio)
                valid_entries.append(entry)
            else:
                num_skipped += 1

        if not audio_list:
            return valid_entries, None, num_skipped

        # Compute mel features (CPU-bound, but overlaps with GPU inference on main thread)
        features = self._extract_features_batch(audio_list)
        return valid_entries, features, num_skipped

    def _run_inference(self, valid_entries, features):
        """Run the appropriate inference mode on pre-computed features."""
        if self.language_detection_only:
            return self._process_lid_batch(valid_entries, features)
        elif self.slice_by_offset:
            return self._process_offset_batch(valid_entries, features)
        else:
            return self._process_transcription_batch(valid_entries, features)

    def _process_single_gpu(self, batches, device_index=0, total_entries=0):
        """Process batches on a single GPU with GPU-based feature extraction.

        Uses the same optimizations as _gpu_worker: soundfile for audio loading
        and PyTorch STFT on GPU for mel features.

        Returns (results_list, total_skipped).
        """
        import time
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        import ctranslate2
        import soundfile as sf
        import torch

        self._load_model(device_index)

        all_results = []
        total_skipped = 0
        num_batches = len(batches)
        if num_batches == 0:
            return all_results, 0

        # ── GPU mel feature extraction setup ──────────────────────────
        device = torch.device(f"cuda:{device_index}")
        n_fft = 400
        hop_length = 160
        N_FRAMES = 3000

        mel_filters_np = self.feature_extractor.mel_filters
        n_freq = mel_filters_np.shape[1]
        mel_filters_gpu = torch.from_numpy(mel_filters_np).float().to(device)
        hann_window = torch.hann_window(n_fft, periodic=True, device=device)

        def _extract_features_gpu(audio_arrays):
            max_len = max(a.shape[0] for a in audio_arrays)
            B = len(audio_arrays)
            batch_np = np.zeros((B, max_len), dtype=np.float32)
            for i, audio in enumerate(audio_arrays):
                batch_np[i, : len(audio)] = audio
            batch_audio = torch.from_numpy(batch_np).to(device)
            stft = torch.stft(
                batch_audio, n_fft=n_fft, hop_length=hop_length,
                window=hann_window, center=False, return_complex=True,
            )
            magnitudes = stft[:, :n_freq, :].abs().pow(2)
            mel_spec = torch.matmul(mel_filters_gpu, magnitudes)
            log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
            log_max = log_spec.flatten(1).max(dim=1).values[:, None, None]
            log_spec = torch.maximum(log_spec, log_max - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            T = log_spec.shape[2]
            if T < N_FRAMES:
                log_spec = torch.nn.functional.pad(log_spec, (0, N_FRAMES - T))
            elif T > N_FRAMES:
                log_spec = log_spec[:, :, :N_FRAMES]
            return log_spec.cpu().numpy()

        # ── Audio loading ─────────────────────────────────────────────
        processor_self = self

        def _load_audio_only(entry):
            try:
                if processor_self.slice_by_offset:
                    audio = processor_self._load_audio(entry)
                    return entry, audio
                filepath = entry[processor_self.audio_filepath_field]
                audio, sr = sf.read(filepath, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return entry, audio
            except Exception:
                if processor_self.skip_corrupted:
                    return entry, None
                raise

        worker_pool = ThreadPoolExecutor(max_workers=12)

        def _prefetch_audio(batch_entries):
            t0 = time.monotonic()
            futs = [worker_pool.submit(_load_audio_only, e) for e in batch_entries]
            results = [f.result() for f in futs]
            valid_entries = []
            audio_list = []
            skipped = 0
            for entry, audio in results:
                if audio is not None:
                    valid_entries.append(entry)
                    audio_list.append(audio)
                else:
                    skipped += 1
            return valid_entries, audio_list, skipped, time.monotonic() - t0

        # ── Prefetch pipeline ─────────────────────────────────────────
        PREFETCH_DEPTH = 3
        prefetch_pool = ThreadPoolExecutor(max_workers=2)
        prefetch_queue = deque()
        for i in range(min(PREFETCH_DEPTH, num_batches)):
            prefetch_queue.append(prefetch_pool.submit(_prefetch_audio, batches[i]))
        next_submit = PREFETCH_DEPTH

        t_start = time.monotonic()
        cum_wait = 0.0
        cum_feat = 0.0
        cum_infer = 0.0

        for batch_idx in range(num_batches):
            t0 = time.monotonic()
            valid_entries, audio_list, skipped, t_audio = prefetch_queue.popleft().result()
            t_wait = time.monotonic() - t0
            cum_wait += t_wait
            total_skipped += skipped

            if next_submit < num_batches:
                prefetch_queue.append(
                    prefetch_pool.submit(_prefetch_audio, batches[next_submit])
                )
                next_submit += 1

            if not valid_entries:
                continue

            t_feat = 0.0
            t_infer = 0.0
            try:
                t0 = time.monotonic()
                features_np = _extract_features_gpu(audio_list)
                features = ctranslate2.StorageView.from_array(features_np)
                t_feat = time.monotonic() - t0
                cum_feat += t_feat

                t0 = time.monotonic()
                results = self._run_inference(valid_entries, features)
                t_infer = time.monotonic() - t0
                cum_infer += t_infer
                all_results.extend(results)
            except Exception:
                if self.skip_corrupted:
                    logger.warning(
                        f"GPU:{device_index} batch {batch_idx} failed. "
                        f"Skipping {len(valid_entries)} entries."
                    )
                    traceback.print_exc()
                    total_skipped += len(valid_entries)
                    continue
                raise

            # Progress
            elapsed = time.monotonic() - t_start
            done_count = len(all_results) + total_skipped
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (total_entries - done_count) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            progress = f"{done_count}/{total_entries}" if total_entries else str(done_count)
            logger.info(
                f"GPU:{device_index} | batch {batch_idx+1}/{num_batches} | {progress} "
                f"| {rate:.0f} items/s | ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s "
                f"| wait={t_wait:.3f}s audio={t_audio:.3f}s "
                f"feat={t_feat:.3f}s infer={t_infer:.3f}s "
                f"| cum: wait={cum_wait:.1f}s feat={cum_feat:.1f}s infer={cum_infer:.1f}s"
            )

        prefetch_pool.shutdown(wait=False)
        worker_pool.shutdown(wait=False)
        return all_results, total_skipped

    def process(self):
        """Main entry point: load model(s), batch-process manifest, write output.

        When num_devices > 1, spawns one process per GPU. Each process loads
        its own model once and pulls batches from a shared queue across all
        manifest chunks, avoiding costly model reloads between chunks.
        """
        self._prepare()

        # Count total entries for progress reporting
        total_entries = 0
        with open(self.input_manifest_file, "r", encoding="utf8") as f:
            for _ in f:
                total_entries += 1
        logger.info(f"Total manifest entries: {total_entries}")

        total_processed = 0
        total_skipped = 0

        if self.num_devices <= 1:
            with open(self.output_manifest_file, "w", encoding="utf8") as fout:
                for chunk in self._chunk_manifest():
                    chunk.sort(key=lambda e: e.get("duration", 0))
                    batches = [
                        chunk[i : i + self.batch_size]
                        for i in range(0, len(chunk), self.batch_size)
                    ]
                    if not batches:
                        continue
                    results, skipped = self._process_single_gpu(batches, device_index=0, total_entries=total_entries)
                    total_skipped += skipped
                    for entry in results:
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_processed += len(results)
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            # Shared queue — all GPUs pull batches from same queue.
            # Faster GPUs naturally grab more work (auto load-balancing).
            # Only manifest entries (file paths) go through the queue — no audio data.
            batch_queue = ctx.Queue()
            result_queue = ctx.Queue()

            entries_per_gpu = total_entries // self.num_devices
            processes = []
            for gpu_id in range(self.num_devices):
                p = ctx.Process(
                    target=_gpu_worker,
                    args=(self, gpu_id, batch_queue, result_queue, entries_per_gpu),
                )
                p.start()
                processes.append(p)

            with open(self.output_manifest_file, "w", encoding="utf8") as fout:
                for chunk in self._chunk_manifest():
                    chunk.sort(key=lambda e: e.get("duration", 0))
                    batches = [
                        chunk[i : i + self.batch_size]
                        for i in range(0, len(chunk), self.batch_size)
                    ]
                    if not batches:
                        continue

                    logger.info(
                        f"Queuing {len(batches)} batches for {self.num_devices} GPUs"
                    )

                    for batch in batches:
                        batch_queue.put(batch)

                # Send sentinels (one per worker)
                for _ in range(self.num_devices):
                    batch_queue.put(None)

                # Collect results
                for p in processes:
                    p.join()

                while not result_queue.empty():
                    gpu_id, results, skipped, error = result_queue.get()
                    if error:
                        logger.error(f"GPU:{gpu_id} failed: {error}")
                        raise error
                    total_skipped += skipped
                    for entry in results:
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_processed += len(results)
                    logger.info(
                        f"GPU:{gpu_id} done: {len(results)} processed, {skipped} skipped"
                    )

        logger.info(
            f"Batched Whisper inference complete: "
            f"{total_processed} processed, {total_skipped} skipped."
        )
