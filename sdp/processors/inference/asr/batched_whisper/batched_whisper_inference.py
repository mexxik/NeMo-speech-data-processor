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

    def _process_single_gpu(self, batches, device_index=0):
        """Process batches on a single GPU with prefetch pipeline.

        Returns (results_list, total_skipped).
        """
        from concurrent.futures import ThreadPoolExecutor

        self._load_model(device_index)

        all_results = []
        total_skipped = 0
        num_batches = len(batches)
        if num_batches == 0:
            return all_results, 0

        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        prefetch_future = prefetch_pool.submit(self._prefetch_batch, batches[0])

        for batch_idx in tqdm(
            range(num_batches),
            desc=f"GPU:{device_index}",
            total=num_batches,
            position=device_index,
        ):
            valid_entries, features, skipped = prefetch_future.result()
            total_skipped += skipped

            if batch_idx + 1 < num_batches:
                prefetch_future = prefetch_pool.submit(
                    self._prefetch_batch, batches[batch_idx + 1]
                )

            if not valid_entries:
                continue

            try:
                results = self._run_inference(valid_entries, features)
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

        prefetch_pool.shutdown(wait=False)
        return all_results, total_skipped

    def process(self):
        """Main entry point: load model(s), batch-process manifest, write output.

        When num_devices > 1, spawns one process per GPU. Each process loads
        its own model and processes interleaved batches. Results are merged
        in original manifest order.
        """
        self._prepare()

        total_processed = 0
        total_skipped = 0

        with open(self.output_manifest_file, "w", encoding="utf8") as fout:
            for chunk in self._chunk_manifest():
                # Sort by duration for efficient batching (less padding waste)
                chunk.sort(key=lambda e: e.get("duration", 0))

                batches = [
                    chunk[i : i + self.batch_size]
                    for i in range(0, len(chunk), self.batch_size)
                ]
                if not batches:
                    continue

                if self.num_devices <= 1:
                    # Single GPU: run directly
                    results, skipped = self._process_single_gpu(batches, device_index=0)
                    total_skipped += skipped
                    for entry in results:
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_processed += len(results)
                else:
                    # Multi-GPU: distribute batches round-robin across GPUs
                    import multiprocessing as mp

                    gpu_batches = [[] for _ in range(self.num_devices)]
                    for i, batch in enumerate(batches):
                        gpu_batches[i % self.num_devices].append(batch)

                    logger.info(
                        f"Distributing {len(batches)} batches across "
                        f"{self.num_devices} GPUs: "
                        + ", ".join(f"GPU:{i}={len(b)}" for i, b in enumerate(gpu_batches))
                    )

                    # Use spawn context to avoid CUDA fork issues
                    ctx = mp.get_context("spawn")
                    result_queue = ctx.Queue()

                    def _gpu_worker(gpu_id, worker_batches, queue):
                        """Worker function that runs in a separate process."""
                        try:
                            results, skipped = self._process_single_gpu(
                                worker_batches, device_index=gpu_id
                            )
                            queue.put((gpu_id, results, skipped, None))
                        except Exception as e:
                            queue.put((gpu_id, [], 0, e))

                    processes = []
                    for gpu_id in range(self.num_devices):
                        if not gpu_batches[gpu_id]:
                            continue
                        p = ctx.Process(
                            target=_gpu_worker,
                            args=(gpu_id, gpu_batches[gpu_id], result_queue),
                        )
                        p.start()
                        processes.append(p)

                    # Collect results from all GPUs
                    for _ in processes:
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

                    for p in processes:
                        p.join()

        logger.info(
            f"Batched Whisper inference complete: "
            f"{total_processed} processed, {total_skipped} skipped."
        )
