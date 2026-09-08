import base64
import os
import gc
import time
import threading
import inspect
import tempfile
from pathlib import Path

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

# ============================================================
# llama-cpp-python (optional, for GGUF loading)
# ============================================================
try:
    from llama_cpp import Llama as LlamaGGUF
    _HAS_LLAMA_CPP = True
except ImportError:
    _HAS_LLAMA_CPP = False

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


# ============================================================
# GGUF helpers
# ============================================================


# Mapping from prefixed model name to its actual full path on disk.
# Populated by _collect_gguf_files() at module load time.
_GGUF_PATH_MAP = {}


def _get_llm_dirs() -> list:
    """
    Return a list of all LLM model directories that exist.
    Searches both 'LLM' and 'llm' subdirectories under models_dir.
    """
    base = folder_paths.models_dir
    dirs = []
    for subdir in ("LLM", "llm"):
        p = os.path.join(base, subdir)
        if os.path.isdir(p):
            dirs.append(p)
    # Deduplicate (in case filesystem is case-insensitive)
    return list(dict.fromkeys(dirs))


def _collect_gguf_files_from_dir(llm_dir: str) -> list:
    """
    Recursively scan an LLM directory and all its subdirectories for .gguf files.
    Returns a sorted list of (display_name, full_path) tuples.
    """
    results = []
    if not os.path.isdir(llm_dir):
        return results

    for root, _dirs, files in os.walk(llm_dir):
        for entry in files:
            if entry.lower().endswith(".gguf") and "mmproj" not in entry.lower():
                full_path = os.path.join(root, entry)
                # Use the full relative path from the LLM root as display name
                rel_path = os.path.relpath(full_path, llm_dir)
                results.append((rel_path, full_path))

    # Sort alphabetically by display name
    results.sort(key=lambda x: x[0])
    return results


def _collect_gguf_files() -> list:
    """
    Scan ALL LLM directories for .gguf files and return a sorted list.
    Searches both 'LLM' and 'llm' directories.
    Also populates _GGUF_PATH_MAP mapping prefixed model names to their full paths.
    """
    all_results = []
    for llm_dir in _get_llm_dirs():
        all_results.extend(_collect_gguf_files_from_dir(llm_dir))
    all_results.sort(key=lambda x: x[0])

    # Build a mapping: "_GGUF__<display_name>" -> <full_path>
    # This ensures _prepare_checkpoint() resolves the correct directory (LLM/ or llm/)
    for display_name, full_path in all_results:
        _GGUF_PATH_MAP[f"_GGUF__{display_name}"] = full_path

    return all_results


def _collect_mmproj_files_from_dir(llm_dir: str) -> list:
    """
    Recursively scan an LLM directory and all its subdirectories for mmproj
    (.gguf) files. Returns a sorted list of (display_name, full_path) tuples.
    """
    results = []
    if not os.path.isdir(llm_dir):
        return results

    for root, _dirs, files in os.walk(llm_dir):
        for entry in files:
            el = entry.lower()
            if el.endswith(".gguf") and "mmproj" in el:
                full_path = os.path.join(root, entry)
                rel_path = os.path.relpath(full_path, llm_dir)
                results.append((rel_path, full_path))

    results.sort(key=lambda x: x[0])
    return results


def _collect_mmproj_files() -> list:
    """
    Scan ALL LLM directories for mmproj (.gguf) files.
    Searches both 'LLM' and 'llm' directories.
    """
    all_results = []
    for llm_dir in _get_llm_dirs():
        all_results.extend(_collect_mmproj_files_from_dir(llm_dir))
    all_results.sort(key=lambda x: x[0])
    return all_results


def _is_gguf_model(model: str) -> bool:
    """Return True when the model name starts with the GGUF prefix."""
    return model.startswith("_GGUF__")


def _extract_gguf_filename(model: str) -> str:
    """Extract the raw GGUF filename from the prefixed model name."""
    return model[len("_GGUF__"):]


def _is_gguf_quant(model: str, quantization: str) -> bool:
    """Return True when the model name indicates GGUF loading.

    Deprecated: equivalent to ``_is_gguf_model``. Kept for API stability
    in case external code calls it.
    """
    return _is_gguf_model(model)


def _detect_gguf_chat_format(model_name: str) -> str:
    """Return the text-only chat_format string for a GGUF model.

    A DEDICATED multimodal `chat_handler` is built by build_qwen_vl_chat_handler()
    for image/video inference, so this string is only used for TEXT-only runs.
    Qwen models are ChatML structurally, so 'chatml' is the correct choice.
    """
    return "chatml"


def build_qwen_vl_chat_handler(mmproj_path: str):
    """Build a multimodal `chat_handler` for GGUF vision inference.

    This is the KEY fix for Qwen3-VL/2.5-VL GGUF in ComfyUI. The upstream
    `llama-cpp-python` `chat_format` registry contains ONLY text handlers
    (llama/llama3/qwen/vicuna/chatml/...); there is no 'llava'/'qwen2_vl'/'qwen3_vl'
    entry, so passing an image via `create_chat_completion` is silently
    dropped and the model only sees the text (canned greeting).

    Instead we import the multimodal handler CLASS directly from
    `llama_cpp.llama_chat_format` and pass it to `Llama(chat_handler=...)`:
      - Prefer Qwen3VLChatHandler (Qwen3-VL models).
      - Fall back to Qwen25VLChatHandler (Qwen2.5-VL); its base class
        (Llava15ChatHandler / LlavaChatHandler) shares the same image
        projection mechanism, so it also works for Qwen3-VL GGUF.
    Each handler's kwargs are filtered with inspect so an unsupported kwarg
    (e.g. 'image_max_tokens') does not raise -- that one belongs on the
    Llama() constructor instead. Returns (handler_cls, accepted_kwargs) or
    (None, None) if no multimodal handler is available.
    """
    if not mmproj_path or not os.path.isfile(mmproj_path):
        print("[QwenVL] WARNING: mmproj missing; cannot build multimodal handler.")
        return None, None
    try:
        import llama_cpp.llama_chat_format as _fmt
    except Exception as exc:
        print(f"[QwenVL] (gguf) cannot import llama_chat_format: {exc}")
        return None, None

    handler_cls = None
    for name in ("Qwen3VLChatHandler", "Qwen25VLChatHandler", "Qwen2VLChatHandler",
                 "Llava15ChatHandler", "LlavaChatHandler"):
        try:
            handler_cls = getattr(_fmt, name)
            break
        except Exception:
            continue

    if handler_cls is None:
        print("[QwenVL] (gguf) NO multimodal handler (Qwen3VL/Qwen25VL) found "
              "in this llama_cpp build; image/video will be ignored.")
        return None, None

    handler_kwargs = {
        "clip_model_path": mmproj_path,
        "image_max_tokens": 4096,
        "force_reasoning": False,
        "verbose": False,
    }
    try:
        accepted = set(inspect.signature(handler_cls.__init__).parameters)
    except Exception:
        accepted = None
    if accepted is not None:
        handler_kwargs = {k: v for k, v in handler_kwargs.items() if k in accepted}
        dropped = {"image_max_tokens", "force_reasoning"} - set(handler_kwargs)
        if dropped:
            print(f"[QwenVL] (gguf) handler {handler_cls.__name__} ignores: "
                  f"{sorted(dropped)} (image_*_tokens belong on Llama()).")

    return handler_cls, handler_kwargs


def _resolve_gguf_path(model: str, llm_dir: str) -> str:
    """
    Resolve the full GGUF file path.

    For auto-discovered models (in _GGUF_PATH_MAP):
        Returns the actual full path directly.
    For fallback (HuggingFace-downloaded models):
        Returns llm_dir / <filename>.
    """
    if model in _GGUF_PATH_MAP:
        return _GGUF_PATH_MAP[model]
    # llm_dir is already the parent directory; use only the bare filename
    raw = _extract_gguf_filename(model)
    filename = os.path.basename(raw)
    return os.path.join(llm_dir, filename)


def tensor_to_pil(image_tensor, batch_index=0):
    """
    Convert a ComfyUI IMAGE tensor to a PIL Image.

    Handles both [0, 1] float and [0, 255] float inputs from ComfyUI.
    Ensures RGB mode (no RGBA/alpha channel) for mmproj compatibility.
    """
    img = image_tensor[batch_index].unsqueeze(0)
    # ComfyUI images can be in [0,1] or [0,255] float32 range.
    # A value slightly above 1.0 (e.g. 1.097) still means a [0,1]-scale
    # image (float overshoot); only clearly large values (>1.5) indicate
    # a [0,255]-scale image. We normalise to [0,1] first, then always
    # scale to 0-255, so we never accidentally treat a bright image as
    # all-black (the old `<= 1.0` check broke on images with max > 1.0).
    if float(img.max()) > 1.5:
        img = img / 255.0
    arr = np.clip(img.cpu().numpy() * 255.0, 0, 255).astype(np.uint8).squeeze()

    # Handle different channel layouts
    if arr.ndim == 2:
        # Grayscale: expand to RGB for mmproj
        pil_img = Image.fromarray(arr, mode='L').convert('RGB')
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            # RGBA: drop alpha channel
            pil_img = Image.fromarray(arr[:, :, :3], mode='RGB')
        else:
            pil_img = Image.fromarray(arr, mode='RGB')
    else:
        raise ValueError(f"Unexpected image tensor shape: {arr.shape}")

    return pil_img


def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 string for OpenAI-compatible API."""
    buffered = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        img.save(buffered, format="PNG")
        buffered.close()
        with open(buffered.name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        try:
            os.unlink(buffered.name)
        except Exception:
            pass
    return ""


def encode_video_to_base64(video_path: str) -> str:
    """Encode a video file to a base64 string for OpenAI-compatible API."""
    if not os.path.isfile(video_path):
        return ""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
            # llama.cpp Llama instances: call .close() to free GPU memory
            if _HAS_LLAMA_CPP and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass
            else:
                _maybe_move_to_cpu(obj)

        entry["resources"].clear()
        _clear_cuda_memory()


# singleton
model_cache = ModelCache()


# ============================================================
# Qwen VL Node
# ============================================================


class QwenVL:
    """
    QwenVL node supporting both HF models and GGUF models with mmproj files.

    - HF models: loaded via transformers (Qwen2.5-VL / Qwen3-VL) with BitsAndBytes quantization
    - GGUF models: loaded via llama-cpp-python with mmproj for vision understanding
    - Supports image input (ComfyUI IMAGE) and video_path for video understanding
    """

    # Preset prompt templates for image description
    PRESET_PROMPTS = {
        "Original": "Describe the image in detail.",
        "Detailed Description": "Please provide a detailed description of this image, including the main subject, setting, colors, lighting, composition, and any notable details. Be thorough and specific about what you see.",
        "Ultra Detailed Description": "Please provide an ultra-detailed description of this image. Cover all aspects: the main subject(s), background, foreground, colors, lighting conditions, shadows, textures, composition, style, mood, atmosphere, and every notable detail. Describe the scene as if painting a picture with words. Include spatial relationships between objects, the time of day suggested, and any subtle details that might be easily missed.",
        "Short Description": "Briefly describe what you see in this image in 1-2 sentences.",
        "Poetic Description": "Describe this image in a poetic, artistic manner. Use vivid imagery, metaphors, and evocative language to capture the mood and essence of the scene.",
        "Technical Analysis": "Analyze this image from a technical perspective: composition (rule of thirds, leading lines, etc.), lighting (direction, quality, color temperature), color palette, depth of field, focus, and overall photographic or artistic technique.",
        "Storytelling": "Tell a story about this image. What is happening? What led up to this moment? What might happen next? Give life to the scene with narrative details about the subject's emotions and surroundings.",
        "Object Detection": "List and describe each distinct object or element visible in this image. Identify the primary subject, secondary elements, and any background objects. Include approximate positions and relative sizes.",
    }

    @classmethod
    def INPUT_TYPES(cls):
        # Auto-scan for GGUF and mmproj files in ALL LLM directories
        gguf_entries = _collect_gguf_files()
        mmproj_entries = _collect_mmproj_files()

        # Build model list: HF models first, then discovered GGUF files
        hf_models = [
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "Qwen3-VL-2B-Thinking",
            "Qwen3-VL-2B-Instruct",
            "Qwen3-VL-4B-Thinking",
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-8B-Thinking",
            "Qwen3-VL-8B-Instruct",
            "Qwen3-VL-32B-Thinking",
            "Qwen3-VL-32B-Instruct",
        ]
        all_models = hf_models + [f"_GGUF__{name}" for name, _ in gguf_entries]
        default_model = hf_models[-1] if hf_models else (
            f"_GGUF__{gguf_entries[0]}" if gguf_entries else "Qwen3-VL-4B-Instruct"
        )

        # Build mmproj list: auto-discovered files or "auto"/"none"
        mmproj_options = ["auto", "none"] + [
            f"_MMPOJ__{name}" for name, _ in mmproj_entries
        ]

        # Preset prompt options for image description style
        preset_options = list(cls.PRESET_PROMPTS.keys())

        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "preset_prompt": (preset_options, {"default": "Detailed Description"}),
                "model": (all_models, {"default": default_model}),
                "quantization": (["none", "4bit", "8bit"],),
                "temperature": ("FLOAT", {"default": 0.7}),
                "max_new_tokens": ("INT", {"default": 512}),
                "seed": ("INT", {"default": -1}),
                "context_size": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 131072,
                    "step": 512,
                }),
                "mmproj": (mmproj_options, {"default": "auto"}),
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
        preset_prompt,
        model,
        quantization,
        temperature,
        max_new_tokens,
        seed,
        context_size=4096,
        mmproj="auto",
        image=None,
        video_path="",
        model_loaded_permanently=False,
        offload_after_used=False,
    ):
        is_gguf = _is_gguf_model(model)

        # DIAGNOSTIC: inspect the raw image tensor BEFORE any conversion.
        # If the tensor is all-zero / black here, the problem is upstream
        # (the image on the wire is already black), not our code.
        if image is not None:
            try:
                import torch
                print(f"[QwenVL] IMAGE TENSOR: dtype={image.dtype} shape={tuple(image.shape)} "
                      f"min={float(image.min()) if image.numel() else 0:.3f} "
                      f"max={float(image.max()) if image.numel() else 0:.3f} "
                      f"mean={float(image.mean()) if image.numel() else 0:.3f}")
                if image.dtype.is_floating_point and image.numel() > 0:
                    zeros = float((image == 0).float().mean())
                    print(f"[QwenVL] FRAZ ZEROS in tensor: {zeros:.3f}")
            except Exception as e:
                print(f"[QwenVL] (diag) could not inspect tensor: {e}")

        # Resolve mmproj path (model-aware: prefer projector next to the model)
        mmproj_path = self._resolve_mmproj(mmproj, model)

        # Validate GGUF prerequisites
        if is_gguf and not _HAS_LLAMA_CPP:
            raise RuntimeError(
                "[QwenVL] GGUF 模型已选择，但 'llama-cpp-python' 未安装。\n"
                "请执行: pip install llama-cpp-python"
            )

        # Determine the model family from the model name
        model_family = self._get_model_family(model)

        # Resolve the preset prompt text (empty string for "Original")
        preset_text = self.PRESET_PROMPTS.get(preset_prompt, "")

        if is_gguf:
            return self._inference_gguf(
                text, preset_text, model, temperature, max_new_tokens, seed, context_size,
                mmproj_path, image, video_path, model_loaded_permanently,
                offload_after_used, model_family
            )
        else:
            return self._inference_hf(
                text, preset_text, model, quantization, temperature, max_new_tokens, seed,
                mmproj_path, image, video_path, model_loaded_permanently,
                offload_after_used, model_family
            )

    def _get_model_family(self, model: str) -> str:
        """Extract the model family name from the model identifier."""
        if _is_gguf_model(model):
            filename = _extract_gguf_filename(model)
            # Try to extract family from filename
            base = Path(filename).stem.lower()
            if "vl" in base:
                return "vl_model"
            return "vl_model"  # default for GGUF
        # For HF models
        if model.startswith("Qwen3"):
            return "vl_model"
        return "vl_model"

    def _resolve_mmproj(self, mmproj: str, model: str = "") -> str:
        """Resolve mmproj (vision projector) file path from user selection.

        Each Qwen-VL model needs its OWN projector. For "auto" we therefore
        prefer the projector that sits in the SAME directory as the selected
        model's GGUF, and only fall back to any mmproj found anywhere if none
        is found next to the model.
        """
        if mmproj == "auto":
            # 1) Prefer the projector that lives next to the chosen model.
            if model in _GGUF_PATH_MAP:
                model_dir = os.path.dirname(_GGUF_PATH_MAP[model])
                for root, _dirs, files in os.walk(model_dir):
                    for entry in files:
                        if entry.lower().endswith(".gguf") and "mmproj" in entry.lower():
                            return os.path.join(root, entry)
            # 2) Fall back to the first mmproj found anywhere.
            all_mmproj = _collect_mmproj_files()
            if all_mmproj:
                return all_mmproj[0][1]  # Already a full path
            return ""
        elif mmproj == "none" or not mmproj:
            return ""
        elif mmproj.startswith("_MMPOJ__"):
            rel = mmproj[len("_MMPOJ__"):]
            # rel_path is relative to <models>/llm (or LLM); also try next to the model.
            for base in (
                os.path.join(folder_paths.models_dir, rel),
                os.path.join(folder_paths.models_dir, "LLM", rel),
                os.path.join(folder_paths.models_dir, "llm", rel),
            ):
                if os.path.isfile(base):
                    return base
            # Fall back to whatever was found during collection.
            all_mmproj = _collect_mmproj_files()
            for name, full in all_mmproj:
                if name == rel:
                    return full
            return os.path.join(folder_paths.models_dir, rel)
        return ""

    def _inference_gguf(
        self,
        text,
        preset_text,
        model,
        temperature,
        max_new_tokens,
        seed,
        context_size,
        mmproj_path,
        image,
        video_path,
        model_loaded_permanently,
        offload_after_used,
        model_family,
    ):
        """Inference using GGUF model + llama-cpp-python.

        VISION (image/video) is supported via a DEDICATED multimodal
        `chat_handler` (see build_qwen_vl_chat_handler), the proven way to
        feed images into Qwen2.5/3-VL GGUF in this llama-cpp build. The plain
        `chat_format`-string path CANNOT render the image and must not be used
        with vision inputs.
        """
        if seed != -1:
            torch.manual_seed(seed)

        # Resolve mmproj + build the multimodal handler (needed for vision).
        handler_cls, handler_kwargs = build_qwen_vl_chat_handler(mmproj_path)

        ckpt = self._prepare_checkpoint(model)
        cache_key = _resolve_gguf_path(model, ckpt)

        # Apply preset prompt by prepending to user's text
        if preset_text:
            if text and text.strip():
                text = f"{preset_text}\n\n{text}"
            else:
                text = preset_text

        # Build the OpenAI-style message content.
        content_parts = [{"type": "text", "text": text}]
        if image is not None:
            img_b64 = pil_to_base64(tensor_to_pil(image))
            if img_b64:
                content_parts.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })

        if video_path:
            video_b64 = encode_video_to_base64(video_path)
            if video_b64:
                content_parts.insert(0, {
                    "type": "video",
                    "video": video_b64
                })

        messages = [{"role": "user", "content": content_parts}]

        n_imgs = sum(1 for p in content_parts if p.get("type") == "image_url")
        has_video = bool(video_path and video_b64)

        def load_model():
            gguf_path = _resolve_gguf_path(model, ckpt)
            if not os.path.exists(gguf_path):
                raise FileNotFoundError(
                    f"[QwenVL] 未找到 GGUF 文件: {gguf_path}\n"
                    f"请将模型文件放入 ComfyUI/models/LLM/ 目录。"
                )

            llm_kwargs = {
                "model_path": gguf_path,
                "n_gpu_layers": -1,
                "n_ctx": context_size,
                "flash_attn": False,
                "verbose": False,
            }

            if handler_cls is not None and handler_kwargs is not None:
                # VISION: inject the multimodal chat_handler object (the KEY fix).
                llm_kwargs["chat_handler"] = handler_cls(**handler_kwargs)
                llm_kwargs["image_min_tokens"] = 1024
                llm_kwargs["image_max_tokens"] = 4096
                print(f"[QwenVL] (gguf) using multimodal handler "
                      f"{handler_cls.__name__} for {n_imgs} image(s)"
                      f"{', +video' if has_video else ''}")
            else:
                # TEXT-only fallback (no multimodal handler / no mmproj).
                llm_kwargs["chat_format"] = _detect_gguf_chat_format(
                    _extract_gguf_filename(model))
                print("[QwenVL] (gguf) NO multimodal handler available; "
                      "loading TEXT-only model (image/video will be ignored).")

            print(f"[QwenVL] LOADING MODEL: {gguf_path}")

            return {"model": LlamaGGUF(**llm_kwargs)}

        cache_quant = "gguf"  # Always "gguf" in the GGUF branch

        model_inst = model_cache.get(
            cache_key, cache_quant, f"{model_family}_{mmproj_path[:50] if mmproj_path else ''}",
            load_model,
            model_loaded_permanently=model_loaded_permanently
        )["model"]

        try:
            response = model_inst.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            return (output,)

        finally:
            model_cache.release(
                cache_key, cache_quant, f"{model_family}_{mmproj_path[:50] if mmproj_path else ''}",
                offload_after_used=offload_after_used
            )

    def _inference_hf(
        self,
        text,
        preset_text,
        model,
        quantization,
        temperature,
        max_new_tokens,
        seed,
        mmproj_path,
        image,
        video_path,
        model_loaded_permanently,
        offload_after_used,
        model_family,
    ):
        """Inference using HF model via transformers."""
        if seed != -1:
            torch.manual_seed(seed)

        ckpt = self._prepare_checkpoint(model)

        # Apply preset prompt by prepending to user's text
        if preset_text:
            if text and text.strip():
                text = f"{preset_text}\n\n{text}"
            else:
                text = preset_text

        # For HF models, we still use processor (not tokenizer)
        processor = model_cache.get(
            ckpt, "none", f"vl_processor_{mmproj_path[:50] if mmproj_path else ''}",
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
            ckpt, quantization, model_family,
            load_model,
            model_loaded_permanently=model_loaded_permanently
        )["model"]

        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

            if image is not None:
                pil_img = tensor_to_pil(image)
                # DIAGNOSTIC: save the converted image so we can confirm it is
                # NOT all-black. If it IS black here, the bug is in the image
                # tensor -> PIL conversion, not in the model.
                diag_path = os.path.join(
                    tempfile.gettempdir(), f"qwenvl_input_{os.getpid()}.png"
                )
                try:
                    pil_img.save(diag_path)
                    print(f"[QwenVL] SAVED converted image to: {diag_path}")
                except Exception as e:
                    print(f"[QwenVL] (diag) failed to save image: {e}")
                messages[0]["content"].insert(0, {
                    "type": "image",
                    "image": pil_img,
                })

            # Handle video if provided
            if video_path and os.path.isfile(video_path):
                messages[0]["content"].insert(0, {
                    "type": "video",
                    "video": video_path,
                })

            text_payload = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = processor(
                text=[text_payload],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(device)

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
                clean_up_tokenization_spaces=True,
            )[0]
            # =========================

            return (output,)

        finally:
            model_cache.release(
                ckpt, quantization, model_family,
                offload_after_used=offload_after_used
            )

    def _prepare_checkpoint(self, model):
        if _is_gguf_model(model):
            # Use the actual directory where the GGUF file was found (LLM/ or llm/)
            if model in _GGUF_PATH_MAP:
                return os.path.dirname(_GGUF_PATH_MAP[model])
            return os.path.join(folder_paths.models_dir, "LLM")
        repo = f"Qwen/{model}"  # HuggingFace organization is "Qwen" (capital Q)
        # Recursively search ALL LLM directories (LLM/ and llm/) for local model at any depth
        for llm_dir in _get_llm_dirs():
            for root, dirs, _files in os.walk(llm_dir):
                if model in dirs:
                    return os.path.join(root, model)
        # Local model not found, download from HuggingFace
        default_path = os.path.join(folder_paths.models_dir, "LLM", model)
        if not os.path.exists(default_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=default_path)
        return default_path

# ============================================================
# Qwen Causal Node
# ============================================================

class Qwen:

    @classmethod
    def INPUT_TYPES(cls):
        # Auto-scan for GGUF files in ALL LLM directories
        gguf_entries = _collect_gguf_files()

        # Build model list: HF models first, then discovered GGUF files
        hf_models = [
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
            "Qwen3-8B-Instruct",
            "Qwen3-4B-Thinking-2507",
            "Qwen3-4B-Instruct-2507",
        ]
        all_models = hf_models + [f"_GGUF__{name}" for name, _ in gguf_entries]
        default_model = hf_models[-1] if hf_models else "Qwen3-4B-Instruct-2507"

        return {
            "required": {
                "system": ("STRING", {"multiline": True}),
                "prompt": ("STRING", {"multiline": True}),
                "model": (all_models, {"default": default_model}),
                "temperature": ("FLOAT", {"default": 0.7}),
                "max_new_tokens": ("INT", {"default": 512}),
                "seed": ("INT", {"default": -1}),
                "context_size": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 131072,
                    "step": 512,
                }),
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
        temperature,
        max_new_tokens,
        seed,
        context_size=4096,
        model_loaded_permanently=False,
        offload_after_used=False,
    ):
        if seed != -1:
            torch.manual_seed(seed)

        is_gguf = _is_gguf_model(model)

        # Validate GGUF prerequisites
        if is_gguf and not _HAS_LLAMA_CPP:
            raise RuntimeError(
                "[Qwen] GGUF 模型已选择，但 'llama-cpp-python' 未安装。\n"
                "请执行: pip install llama-cpp-python"
            )

        ckpt = self._prepare_checkpoint(model)

        # For GGUF, llama.cpp handles chat templates internally; no tokenizer needed
        if is_gguf:
            tokenizer = None
        else:
            tokenizer = model_cache.get(
                ckpt, "none", "causal_tokenizer",
                lambda: {"tokenizer": AutoTokenizer.from_pretrained(ckpt)}
            )["tokenizer"]

        # Cache key: unique per resolved path
        cache_key = _resolve_gguf_path(model, ckpt) if is_gguf else ckpt

        def load_model():
            if is_gguf:
                gguf_path = _resolve_gguf_path(model, ckpt)
                if not os.path.exists(gguf_path):
                    raise FileNotFoundError(
                        f"[Qwen] 未找到 GGUF 文件: {gguf_path}\n"
                        f"请将模型文件放入 ComfyUI/models/LLM/ 目录。"
                    )
                # Detect chat_format for Qwen models
                chat_format = _detect_gguf_chat_format(_extract_gguf_filename(model))
                return {
                    "model": LlamaGGUF(
                        model_path=gguf_path,
                        n_gpu_layers=-1,        # all layers to GPU (critical for 16GB VRAM)
                        n_ctx=context_size,     # context window size
                        flash_attn=False,       # required for Qwen3 on CUDA
                        verbose=False,
                        chat_format=chat_format,
                    )
                }
            else:
                return {
                    "model": AutoModelForCausalLM.from_pretrained(
                        ckpt,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                        quantization_config=None,
                    )
                }

        cache_quant = "gguf" if is_gguf else "none"

        model_inst = model_cache.get(
            cache_key, cache_quant, "causal_model",
            load_model,
            model_loaded_permanently=model_loaded_permanently
        )["model"]

        try:
            if is_gguf:
                # llama.cpp uses OpenAI-compatible chat completions
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                response = model_inst.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                output = response["choices"][0]["message"]["content"]
            else:
                # Original transformers-based inference
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(text, return_tensors="pt").to(device)
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
                    clean_up_tokenization_spaces=True,
                )

            return (output,)

        finally:
            model_cache.release(
                cache_key, cache_quant, "causal_model",
                offload_after_used=offload_after_used
            )

    def _prepare_checkpoint(self, model):
        """Resolve the base directory for the model. For GGUF files, returns actual dir."""
        if _is_gguf_model(model):
            if model in _GGUF_PATH_MAP:
                return os.path.dirname(_GGUF_PATH_MAP[model])
            return os.path.join(folder_paths.models_dir, "LLM")
        repo = f"Qwen/{model}"  # HuggingFace organization is "Qwen" (capital Q)
        # Recursively search ALL LLM directories (LLM/ and llm/) for local model at any depth
        for llm_dir in _get_llm_dirs():
            for root, dirs, _files in os.walk(llm_dir):
                if model in dirs:
                    return os.path.join(root, model)
        # Local model not found, download from HuggingFace
        default_path = os.path.join(folder_paths.models_dir, "LLM", model)
        if not os.path.exists(default_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo, local_dir=default_path)
        return default_path


# ============================================================
# ComfyUI Node Registration
# ============================================================

NODE_CLASS_MAPPINGS = {
    "qwen_VL_model": QwenVL,
    "qwen_chat_model": Qwen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "qwen_VL_model": "qwen_VL_model",
    "qwen_chat_model": "qwen_chat_model",
}
