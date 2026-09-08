# ComfyUI Qwen chat models node

**语言**: [English](README.md) | **中文**

本项目为 ComfyUI 自定义节点：在工作流里直接调用 **Qwen 文本模型**（Qwen2.5 / Qwen3）与 **Qwen 多模态模型**（Qwen2.5-VL / Qwen3-VL）进行生成。

- 支持纯文本对话与多模态对话（文本 + 可选图片输入）
- 支持 `none / 4bit / 8bit` 量化以降低显存占用（依赖 bitsandbytes）
- 内置 **模型缓存管理**：可选择“常驻显存”或“用完卸载/清理显存”

> 模型会在首次运行节点时自动从 Hugging Face 下载到本地（见下方 Model Storage）。推荐从本地加载。

## Sample Workflows 
### default-max-loaded-models=2
- basic use of nodes:
包含了基本节点的使用方式
[`workflow_example/basic_flow.json`](workflow_example/basic_flow.json)
![basic use of nodes](workflow_example/example1-basic_flow.jpg)
- advanced use of cache control:你可以看到，当我们开启了预加载模式时，再次调用相同的节点会显著加速。 [`workflow_example/advanced_cache_manage.json`](workflow_example/advanced_cache_manage.json)

![advanced use of cache control](workflow_example/example2-advanced-cache-manage_flow.jpg)

## Installation

你可以手动安装：

1. 克隆仓库：

   ```bash
   git clone https://github.com/ConstantlyGrowup/ComfyUI_Qwen_chat_models.git
   ```

2. 进入目录：

   ```bash
   cd ComfyUI_Qwen_chat_models
   ```

3. 安装依赖（建议在 ComfyUI 的虚拟环境中执行）：

   ```bash
   pip install -r requirements.txt
   ```

4. 将本仓库放入 ComfyUI 的自定义节点目录：

   - 推荐路径：`ComfyUI/custom_nodes/ComfyUI_Qwen_chat_models`
   - 或创建 symlink/junction 到该目录
   - Windows 示例：`D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_Qwen_chat_models`

5. 重启 ComfyUI，在节点分类 `Comfyui_Qwen` 下即可看到本项目节点。

## Supported Nodes

- **QwenVL node (Qwen2.5-VL / Qwen3-VL)**：多模态对话生成（文本 + 可选图片输入）。
- **Qwen node (Qwen2.5 / Qwen3)**：纯文本对话生成（system + user prompt）。

### QwenVL node (multimodal)

- **Inputs (required)**:
  - **text**：用户文本提示（STRING, multiline）
  - **model**：选择 VL checkpoint（下拉列表）
  - **quantization**：`none / 4bit / 8bit`
  - **temperature**：采样温度（FLOAT）
  - **max_new_tokens**：最大生成 token（INT）
  - **seed**：随机种子；`-1` 表示不设置（INT）
- **Inputs (optional)**:
  - **image**：ComfyUI 的 `IMAGE`（传入后会作为对话中的 image content）
  - **video_path**：预留字段（当前实现不读取该路径；后续可扩展）
  - **model_loaded_permanently**：是否将“模型”常驻缓存（BOOLEAN）
  - **offload_after_used**：是否在本次推理结束后卸载模型并清理显存（BOOLEAN）
- **Outputs**:
  - **STRING**：模型回复文本

### Qwen node (text-only)

- **Inputs (required)**:
  - **system**：system 提示词（STRING, multiline）
  - **prompt**：user 提示词（STRING, multiline）
  - **model**：选择文本 checkpoint（下拉列表）
  - **quantization**：`none / 4bit / 8bit`
  - **temperature / max_new_tokens / seed**：同上
- **Inputs (optional)**:
  - **model_loaded_permanently**：是否将“模型”常驻缓存（BOOLEAN）
  - **offload_after_used**：是否在本次推理结束后卸载模型并清理显存（BOOLEAN）
- **Outputs**:
  - **STRING**：模型回复文本

### Cache / VRAM management（重要）

本项目内置一个全局 `ModelCache`，用来在多次调用节点时复用模型、减少重复加载开销，并提供显存释放能力：

- **常驻 (Pinned)**：勾选 `model_loaded_permanently=True` 时，该模型会被 pin 在缓存里，不会被 LRU 淘汰。
- **用完卸载**：勾选 `offload_after_used=True` 时，本次推理结束会主动把该模型从缓存中移除，并尝试清理显存。
- **LRU 淘汰**：当加载的模型数量超过阈值，会按最近最少使用淘汰 *非 pinned* 的模型（只对 `*_model` 生效）。

你可以通过环境变量调整缓存上限：

- `QWEN_MAX_LOADED_MODELS`（默认 `2`）：允许同时保留的模型数量（只统计 `*_model`，processor/tokenizer 不计入且会常驻缓存）。这个限制可以在node.py里面找到。

> 说明：`processor` / `tokenizer` 会被永久缓存（它们通常很轻量）；真正占用显存/内存大头的是 `*_model`。

## Model Storage

下载的模型会存放在：

- `ComfyUI/models/LLM/<model_name>/`

节点会在首次使用某个模型时自动下载（等价于 Hugging Face `snapshot_download` 到上述目录）。

### Supported model names (as of current node lists)

- **VL (QwenVL)**:
  - `Qwen2.5-VL-3B-Instruct`
  - `Qwen2.5-VL-7B-Instruct`
  - `Qwen3-VL-2B-Thinking`, `Qwen3-VL-2B-Instruct`
  - `Qwen3-VL-4B-Thinking`, `Qwen3-VL-4B-Instruct`
  - `Qwen3-VL-8B-Thinking`, `Qwen3-VL-8B-Instruct`
  - `Qwen3-VL-32B-Thinking`, `Qwen3-VL-32B-Instruct`
- **Text (Qwen)**:
  - `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Qwen2.5-14B-Instruct`, `Qwen2.5-32B-Instruct`
  - `Qwen3-8B-Instruct`
  - `Qwen3-4B-Thinking-2507`, `Qwen3-4B-Instruct-2507`

## GGUF 模型支持

从版本 `0.2.0 起，本插件支持通过 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 加载 **GGUF** 格式模型（如 Qwen3-14B-Uncensored 文本，以及 Qwen2.5-VL / Qwen3-VL 多模态）。这非常适合在显存有限的消费级显卡上运行大型模型。

> **多模态（图像 / 视频）的重要前提：需要一个带「多模态 handler）的 GGUF 构建。** 标准 PyPI 版 `llama-cpp-python` 常常只提供**纯文本的 chat handler，这种情况下 `create_chat_completion` 会**静默丢弃图像，只返回模板化问候语。因此需要暴露`Qwen3VLChatHandler` / `Qwen25VLChatHandler` 的构建。详见下方【前置条件】。

### 前置条件

**仅纯文本 GGUF（Qwen 文本节点）只需普通安装：

```bash
pip install llama-cpp-python
```

**多模态（图像 / 视频）需要一个暴露「多模态 handler的构建（`Qwen3VLChatHandler` 或 `Qwen25VLChatHandler`——标准 PyPI 版常只提供**纯文本的 chat handler，此时 `create_chat_completion` 会静默丢弃图像、返回模板化问候语。

验证你的构建是否支持多模态：

```bash
python3 -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler, Qwen25VLChatHandler; print('vision OK')"
```

若失败，请安装支持多模态的构建（如 `JamePeng/llama-cpp-python` 的 wheel，参见 [AILAB 的安装文档](https://github.com/1038lab/ComfyUI-QwenVL/blob/main/docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md)。**装到你的 ComfyUI Python 环境**里，再重新验证。

如需 CUDA 加速（Windows），可能需要使用 CUDA 编译：

```bash
CMAKE_ARGS=”-DGGML_CUDA=on -DLLAMA_CURL=OFF” pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

### 使用方法

1. 从 HuggingFace 下载任意 GGUF 模型文件（如 `Qwen3.5-9B-UD-Q8_K_XL.gguf`、`Qwen3.5-9B-Q8_0.gguf`、`Qwen3-14B-Uncensored.Q6_K.gguf`）。
2. 将文件直接放入 **`ComfyUI/models/LLM/`** 目录（不需要子目录）。
3. 重启 ComfyUI（或在节点中切换模型后切回来即可重新扫描）。
4. 在节点中，你的 GGUF 文件会**自动出现在模型下拉框**中——无需手动填写文件路径。

### 上下文大小（context_size）

`context_size` 参数（默认 **4096**）控制模型的上下文窗口（`n_ctx`）。需要更长对话时可以提高它，但注意：
- 上下文越大，显存占用越高（KV 缓存随上下文线性增长）
- 对于 14B Q6_K 模型在 16GB 显存上，建议 context ≤ 8192 以避免 OOM
- 推荐值：2048（速度快、短对话）、4096（平衡）、8192（较长对话，需额外 ~2GB 显存用于 KV 缓存）

### 16GB 显存显卡推荐配置（如 RTX 4070 Ti Super）

| 模型 | 量化 | 显存占用 |
|---|---|---|
| Qwen3-14B-Uncensored | Q6_K | ~12.2 GB 权重（8K context 下约 15.1 GB） |
| Qwen3-14B-Uncensored | Q5_K_M | ~10.5 GB 权重（8K context 下约 13.3 GB） |

> **注意**：使用 GGUF 时，`temperature` 和 `max_new_tokens` 会传递给 llama.cpp 的 `create_chat_completion`。`seed` 参数对 GGUF 模型无效。

## 常见问题

- **勾选了 model_loaded_permanently，然后报错 “Cannot load pinned model”**：
  - 说明 pinned 模型数量超过 `QWEN_MAX_LOADED_MODELS`（pinned 默认 2）
  - 取消部分节点的 `model_loaded_permanently`，或提高环境变量 `QWEN_MAX_LOADED_MODELS`

- **我的 GGUF 文件没有出现在模型下拉框中**：
  - 确认文件直接在 `ComfyUI/models/LLM/` 下（不在子目录里）
  - 确认文件名以 `.gguf` 结尾（不区分大小写）
  - 重启 ComfyUI

- **GGUF 模型不理解图像 / 视频：
  - **你需要一个「支持多模态的 `llama-cpp-python` 构建。** 普通 PyPI 版只有**文本的 handler，`create_chat_completion` 会静默丢弃图像、返回模板化问候语。请先在【前置条件】**验证你的构建，若不支持，则安装暴露 `Qwen3VLChatHandler` / `Qwen25VLChatHandler 的构建并重新验证。
  - 确保已加载 mmproj（设置 `mmproj: auto` 或选择具体 mmproj）
  - 确认 mmproj 文件在 `ComfyUI/models/LLM/` 中，且文件名包含 'mmproj'
  - mmproj 必须与 GGUF 模型匹配（如 Qwen2.5-VL mmproj 对应 Qwen2.5-VL 的 GGUF）

- **CUDA OOM / 显存占用一直不降**：
  - 勾选 `offload_after_used=True` 让节点在推理结束后卸载模型并执行显存清理
  - 减小 `context_size` 或 `max_new_tokens`
  - 尝试更小的量化（如 Q6_K → Q5_K_M）


