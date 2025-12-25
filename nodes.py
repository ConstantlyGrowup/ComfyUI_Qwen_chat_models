import os
import gc
import time
import threading
import inspect
import uuid
import subprocess

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import folder_paths

try:
    import comfy.model_management as comfy_mm
except ImportError:
    comfy_mm = None


# ============================================================
# Utils
# ============================================================

def _maybe_move_to_cpu(obj):
    if obj is None:
        return
    try:
        obj.to("cpu")
    except Exception:
        pass


def _clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if comfy_mm:
        try:
            soft_empty = getattr(comfy_mm, "soft_empty_cache", None)
            if callable(soft_empty):
                params = inspect.signature(soft_empty).parameters
                soft_empty(force=True) if "force" in params else soft_empty()
        except Exception:
            pass


def tensor_to_pil(image_tensor, batch_index=0):
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    arr = 255.0 * image_tensor.cpu().numpy()
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8).squeeze())


# ============================================================
# Model Cache (Pinned + LRU, NO refcount)
# ============================================================

class ModelCache:
    """
    - Only *_model families are counted toward max_loaded_models
    - processor / tokenizer are lightweight and cached permanently
    """

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self.max_loaded_models = int(os.environ.get("QWEN_MAX_LOADED_MODELS", "2"))

    def _make_key(self, ckpt, quant, family):
        return f"{ckpt}|{quant}|{family}"

    def _is_model_family(self, family: str) -> bool:
        return family.endswith("_model")

    def _pinned_model_count(self):
        return sum(
            1 for v in self._cache.values()
            if v["pinned"] and self._is_model_family(v["family"])
        )

    # -------------------------
    # Public API
    # -------------------------

    def get(
        self,
        checkpoint_dir,
        quantization,
        family,
        loader_func,
        *,
        model_loaded_permanently=False,
    ):
        key = self._make_key(checkpoint_dir, quantization, family)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry["last_used"] = time.time()
                return entry["resources"]

            pinned = bool(model_loaded_permanently) and self._is_model_family(family)

            if pinned:
                if self._pinned_model_count() + 1 > self.max_loaded_models:
                    pinned_models = [
                        k for k, v in self._cache.items()
                        if v["pinned"] and self._is_model_family(v["family"])
                    ]
                    raise RuntimeError(
                        "[ModelCache] Cannot load pinned model.\n"
                        f"Trying to pin: {key}\n"
                        f"Pinned models: {pinned_models}\n"
                        f"max_loaded_models={self.max_loaded_models}\n"
                        "Unpin a model or increase QWEN_MAX_LOADED_MODELS."
                    )

            resources = loader_func()

            self._cache[key] = {
                "resources": resources,
                "family": family,
                "pinned": pinned,
                "last_used": time.time(),
            }

            self._evict_if_needed()
            return resources

    def release(
        self,
        checkpoint_dir,
        quantization,
        family,
        *,
        offload_after_used=False,
    ):
        if not offload_after_used:
            return

        key = self._make_key(checkpoint_dir, quantization, family)
        with self._lock:
            self._unload_key(key)

    # -------------------------
    # Eviction
    # -------------------------

    def _evict_if_needed(self):
        """
        Only evict *_model entries.
        """
        while self._loaded_model_count() > self.max_loaded_models:
            candidates = [
                (k, v) for k, v in self._cache.items()
                if self._is_model_family(v["family"]) and not v["pinned"]
            ]
            if not candidates:
                break

            candidates.sort(key=lambda kv: kv[1]["last_used"])
            self._unload_key(candidates[0][0])

    def _loaded_model_count(self):
        return sum(
            1 for v in self._cache.values()
            if self._is_model_family(v["family"])
        )

    def _unload_key(self, key):
        entry = self._cache.pop(key, None)
        if not entry:
            return

        for obj in entry["resources"].values():
            _maybe_move_to_cpu(obj)

        entry["resources"].clear()
        _clear_cuda_memory()


# singleton
model_cache = ModelCache()


# ============================================================
# Qwen VL Node
# ============================================================

# =========================
# Qwen VL Node (FIXED OUTPUT)
# =========================

class QwenVL:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                 "model": (
                    [
                        "Qwen2.5-VL-3B-Instruct",
                        "Qwen2.5-VL-7B-Instruct",
                        "Qwen3-VL-2B-Thinking",
                        "Qwen3-VL-2B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-32B-Thinking",
                        "Qwen3-VL-32B-Instruct"
                    ],
                    {"default": "Qwen3-VL-4B-Instruct"},
                ),
                "quantization": (["none", "4bit", "8bit"],),
                "temperature": ("FLOAT", {"default": 0.7}),
                "max_new_tokens": ("INT", {"default": 512}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "model_loaded_permanently": ("BOOLEAN", {"default": False}),
                "offload_after_used": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen"

    def inference(
        self,
        text,
        model,
        quantization,
        temperature,
        max_new_tokens,
        seed,
        image=None,
        video_path="",
        model_loaded_permanently=False,
        offload_after_used=False,
    ):
        if seed != -1:
            torch.manual_seed(seed)

        ckpt = self._prepare_checkpoint(model)

        processor = model_cache.get(
            ckpt, "none", "vl_processor",
            lambda: {"processor": AutoProcessor.from_pretrained(ckpt)}
        )["processor"]

        def load_model():
            qc = None
            if quantization == "4bit":
                qc = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                qc = BitsAndBytesConfig(load_in_8bit=True)

            cls = Qwen3VLForConditionalGeneration if model.startswith("Qwen3") \
                else Qwen2_5_VLForConditionalGeneration

            return {
                "model": cls.from_pretrained(
                    ckpt,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    quantization_config=qc,
                )
            }

        model_inst = model_cache.get(
            ckpt, quantization, "vl_model",
            load_model,
            model_loaded_permanently=model_loaded_permanently
        )["model"]

        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

            if image is not None:
                messages[0]["content"].insert(0, {
                    "type": "image",
                    "image": tensor_to_pil(image),
                })

            text_payload = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_payload],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to("cuda")

            # ========= FIXED =========
            input_ids = inputs.input_ids

            generated_ids = model_inst.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]

            output = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            # =========================

            return (output,)

        finally:
            model_cache.release(
                ckpt, quantization, "vl_model",
                offload_after_used=offload_after_used
            )


    def _prepare_checkpoint(self, model):
        repo = f"qwen/{model}"
        path = os.path.join(folder_paths.models_dir, "LLM", model)
        if not os.path.exists(path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=path)
        return path


# ============================================================
# Qwen Causal Node
# ============================================================

# =========================
# Qwen Causal Node (FIXED)
# =========================

class Qwen:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": ("STRING", {"multiline": True}),
                "prompt": ("STRING", {"multiline": True}),
               "model": (
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-7B-Instruct",
                        "Qwen2.5-14B-Instruct",
                        "Qwen2.5-32B-Instruct",
                        "Qwen3-8B-Instruct",
                        "Qwen3-4B-Thinking-2507",
                        "Qwen3-4B-Instruct-2507"
                    ],
                    {"default": "Qwen3-4B-Instruct-2507"},
                ),
                "quantization": (["none", "4bit", "8bit"],),
                "temperature": ("FLOAT", {"default": 0.7}),
                "max_new_tokens": ("INT", {"default": 512}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "model_loaded_permanently": ("BOOLEAN", {"default": False}),
                "offload_after_used": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen"

    def inference(
        self,
        system,
        prompt,
        model,
        quantization,
        temperature,
        max_new_tokens,
        seed,
        model_loaded_permanently=False,
        offload_after_used=False,
    ):
        if seed != -1:
            torch.manual_seed(seed)

        ckpt = self._prepare_checkpoint(model)

        tokenizer = model_cache.get(
            ckpt, "none", "causal_tokenizer",
            lambda: {"tokenizer": AutoTokenizer.from_pretrained(ckpt)}
        )["tokenizer"]

        def load_model():
            qc = None
            if quantization == "4bit":
                qc = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                qc = BitsAndBytesConfig(load_in_8bit=True)

            return {
                "model": AutoModelForCausalLM.from_pretrained(
                    ckpt,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    quantization_config=qc,
                )
            }

        model_inst = model_cache.get(
            ckpt, quantization, "causal_model",
            load_model,
            model_loaded_permanently=model_loaded_permanently
        )["model"]

        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids

            generated_ids = model_inst.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]

            output = tokenizer.decode(
                generated_ids_trimmed[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return (output,)

        finally:
            model_cache.release(
                ckpt, quantization, "causal_model",
                offload_after_used=offload_after_used
            )


    def _prepare_checkpoint(self, model):
        repo = f"qwen/{model}"
        path = os.path.join(folder_paths.models_dir, "LLM", model)
        if not os.path.exists(path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=path)
        return path
