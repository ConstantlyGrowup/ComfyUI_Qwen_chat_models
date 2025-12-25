from .nodes import Qwen, QwenVL

# Only expose 2 nodes:
# - qwen4chat: text-only / causal chat models (Qwen2.5, Qwen3, etc.)
# - qwen4VL: vision-language models (Qwen2.5-VL, Qwen3-VL, etc.)
#
# Both nodes share the same internal cache (LRU + pin/unpin + manual release)
# implemented in `nodes.py`, so multiple nodes in one workflow won't reload models.
NODE_CLASS_MAPPINGS = {
    "qwen_chat_model": Qwen,
    "qwen_VL_model": QwenVL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "qwen_chat_model": "qwen_chat_model",
    "qwen_VL_model": "qwen_VL_model",
}
