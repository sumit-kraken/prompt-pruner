import os
import sys
import time
from typing import Dict, List, Any, Optional
import numpy as np

# Configure environment before transformers imports to avoid Windows SAC dll blocks
os.environ["USE_TORCH"] = "0"
os.environ["TRANSFORMERS_NO_TORCH"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Dynamically add Ollama / NVIDIA CUDA dll path if present on Windows
ollama_cuda_paths = [
    r"C:\Users\Shloka\AppData\Local\Programs\Ollama\lib\ollama\cuda_v13",
    r"C:\Users\Shloka\AppData\Local\Programs\Ollama\lib\ollama\cuda_v12",
]
for p in ollama_cuda_paths:
    if os.path.exists(p) and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort
from transformers import AutoTokenizer


class TokenPruner:
    """
    LLMLingua-2 Token Pruning engine using ONNX Runtime.
    Evaluates token retention probabilities and prunes non-essential tokens.
    """

    def __init__(self, model_dir: str = "backend/llmlingua2_onnx"):
        # Resolve model directory relative to repo root or current file
        candidates = [
            model_dir,
            os.path.join(os.path.dirname(__file__), "..", model_dir),
            os.path.join(os.path.dirname(__file__), "llmlingua2_onnx"),
            os.path.abspath(model_dir),
        ]
        resolved_dir = None
        for cand in candidates:
            if os.path.exists(cand) and (
                os.path.exists(os.path.join(cand, "model.onnx"))
                or os.path.exists(os.path.join(cand, "onnx", "model.onnx"))
            ):
                resolved_dir = os.path.abspath(cand)
                break

        if not resolved_dir:
            resolved_dir = os.path.abspath(model_dir)

        self.model_dir = resolved_dir

        # Locate ONNX model file
        model_file = os.path.join(self.model_dir, "model.onnx")
        if not os.path.exists(model_file):
            alt_model = os.path.join(self.model_dir, "onnx", "model.onnx")
            if os.path.exists(alt_model):
                model_file = alt_model

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"ONNX model file not found in {self.model_dir}")

        # Initialize Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        except Exception as e:
            # Fallback to tokenizers Rust library directly if transformers fails
            from tokenizers import Tokenizer
            tok_json = os.path.join(self.model_dir, "tokenizer.json")
            self._raw_tokenizer = Tokenizer.from_file(tok_json)
            self.tokenizer = None

        # Setup ONNX Session with CUDA Execution Provider and CPU Fallback
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, os.cpu_count() or 4)

        available_providers = ort.get_available_providers()
        preferred_providers = []
        if "CUDAExecutionProvider" in available_providers:
            preferred_providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available_providers:
            preferred_providers.append("CPUExecutionProvider")
        if not preferred_providers:
            preferred_providers = available_providers

        try:
            self.session = ort.InferenceSession(
                model_file, sess_options=sess_options, providers=preferred_providers
            )
            self.active_provider = self.session.get_providers()[0]
        except Exception as err:
            # Fallback to CPU strictly if provider initialization fails
            print(f"[TokenPruner] Warning: Provider init failed ({err}), falling back to CPUExecutionProvider.")
            self.session = ort.InferenceSession(
                model_file, sess_options=sess_options, providers=["CPUExecutionProvider"]
            )
            self.active_provider = "CPUExecutionProvider"

        print(f"[TokenPruner] Initialized successfully with provider: {self.active_provider}")

        # Cache model input/output names
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Warm up session with dummy pass
        try:
            dummy_ids = np.zeros((1, 16), dtype=np.int64)
            dummy_mask = np.ones((1, 16), dtype=np.int64)
            dummy_feed = {"input_ids": dummy_ids, "attention_mask": dummy_mask}
            if "token_type_ids" in self.input_names:
                dummy_feed["token_type_ids"] = np.zeros((1, 16), dtype=np.int64)
            self.session.run(None, dummy_feed)
        except Exception:
            pass

    def _tokenize(self, text: str):
        """Tokenize text and return encodings with offset mapping."""
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                add_special_tokens=True,
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                "offsets": enc["offset_mapping"],
                "special_tokens_mask": enc.get("special_tokens_mask", [0] * len(enc["input_ids"])),
            }
        else:
            enc = self._raw_tokenizer.encode(text)
            spec_mask = [1 if t in ("[CLS]", "[SEP]", "[PAD]") else 0 for t in enc.tokens]
            return {
                "input_ids": enc.ids,
                "attention_mask": enc.attention_mask,
                "token_type_ids": enc.type_ids,
                "offsets": enc.offsets,
                "special_tokens_mask": spec_mask,
            }

    def prune(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Prune input text based on token importance classification.

        Args:
            text: Input text/prompt to prune.
            threshold: Probability cutoff (0.0 to 1.0) for keeping tokens.

        Returns:
            Dictionary matching the specified schema:
            {
                "tokens": [{"word": str, "score": float, "is_kept": bool, "start": int, "end": int}],
                "compressed_text": str,
                "total_tokens": int,
                "pruned_tokens": int,
                "saved_pct": float,
                "latency_ms": float,
                "provider": str
            }
        """
        if not text or not text.strip():
            return {
                "tokens": [],
                "compressed_text": "",
                "total_tokens": 0,
                "pruned_tokens": 0,
                "saved_pct": 0.0,
                "latency_ms": 0.0,
                "provider": self.active_provider,
            }

        start_time = time.perf_counter()

        # Tokenize with offset mapping
        enc = self._tokenize(text)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        token_type_ids = enc["token_type_ids"]
        offsets = enc["offsets"]
        special_mask = enc["special_tokens_mask"]

        seq_len = len(input_ids)
        max_seq_len = 512

        # Process in chunks if sequence length exceeds BERT limit (512)
        if seq_len <= max_seq_len:
            feed = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "attention_mask": np.array([attention_mask], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array([token_type_ids], dtype=np.int64)

            outputs = self.session.run(None, feed)
            logits = outputs[0][0]  # Shape: [seq_len, 2]

            # Softmax calculation for Class 1 (preserve probability)
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            p_keep_all = probs[:, 1]
        else:
            # Chunking with window of 510 tokens (+CLS and +SEP)
            chunk_size = 510
            p_keep_all = np.zeros(seq_len, dtype=np.float32)
            # Find cls and sep token ids
            cls_id = input_ids[0] if seq_len > 0 else 101
            sep_id = input_ids[-1] if seq_len > 0 else 102

            # Strip original CLS/SEP for internal chunks
            core_ids = input_ids[1:-1] if seq_len > 2 else input_ids
            for c_start in range(0, len(core_ids), chunk_size):
                chunk_slice = core_ids[c_start : c_start + chunk_size]
                sub_ids = [cls_id] + chunk_slice + [sep_id]
                sub_mask = [1] * len(sub_ids)
                sub_type = [0] * len(sub_ids)

                feed = {
                    "input_ids": np.array([sub_ids], dtype=np.int64),
                    "attention_mask": np.array([sub_mask], dtype=np.int64),
                }
                if "token_type_ids" in self.input_names:
                    feed["token_type_ids"] = np.array([sub_type], dtype=np.int64)

                outputs = self.session.run(None, feed)
                sub_logits = outputs[0][0]
                exp_l = np.exp(sub_logits - np.max(sub_logits, axis=-1, keepdims=True))
                sub_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
                # Map back to main indices (skipping CLS and SEP of the chunk)
                p_keep_all[1 + c_start : 1 + c_start + len(chunk_slice)] = sub_probs[1:-1, 1]

        # Extract tokens and evaluate keep/prune decision
        tokens_list = []
        for i in range(seq_len):
            s, e = offsets[i]
            # Skip special tokens ([CLS], [SEP], [PAD]) or empty offset spans
            if s >= e or special_mask[i] == 1:
                continue

            score = float(p_keep_all[i])
            is_kept = bool(score >= threshold)
            word = text[s:e]

            tokens_list.append({
                "word": word,
                "score": round(score, 4),
                "is_kept": is_kept,
                "start": int(s),
                "end": int(e),
            })

        # Calculate metrics
        total_tokens = len(tokens_list)
        pruned_tokens = sum(1 for t in tokens_list if not t["is_kept"])
        saved_pct = round((pruned_tokens / total_tokens * 100), 2) if total_tokens > 0 else 0.0

        # Construct compressed text from kept spans
        compressed_parts = []
        last_end = -1
        for t in tokens_list:
            if t["is_kept"]:
                s, e = t["start"], t["end"]
                if last_end != -1:
                    between = text[last_end:s]
                    if "\n" in between:
                        compressed_parts.append("\n" * min(2, between.count("\n")))
                    elif between.strip() == "" and len(between) > 0:
                        compressed_parts.append(" ")
                    elif s > last_end:
                        compressed_parts.append(" ")
                compressed_parts.append(text[s:e])
                last_end = e

        compressed_text = "".join(compressed_parts).strip()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "tokens": tokens_list,
            "compressed_text": compressed_text,
            "total_tokens": total_tokens,
            "pruned_tokens": pruned_tokens,
            "saved_pct": saved_pct,
            "latency_ms": latency_ms,
            "provider": self.active_provider,
        }


if __name__ == "__main__":
    pruner = TokenPruner()
    sample_text = (
        "2026-10-29 12:00:00 [ERROR] NullPointerException at "
        "com.example.service.AuthService.login(AuthService.java:42) user_id=9812 - "
        "Failed to authenticate user due to unexpected database timeout during credential verification."
    )
    result = pruner.prune(sample_text, threshold=0.5)
    print("--- PRUNER VERIFICATION ---")
    print(f"Provider: {result['provider']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Pruned tokens: {result['pruned_tokens']}")
    print(f"Saved %: {result['saved_pct']}%")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"Compressed text: {result['compressed_text']}")
