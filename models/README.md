# Built-in Models

Pre-built BPE tokenizer models in `.tbm` format, shipped directly with the package — no network required.

## Quick Start

```python
from tinybpe import Tokenizer, list_models

# See what's available
print(list_models())

# Load in one line
tok = Tokenizer.from_pretrained("cl100k_base")
ids = tok.encode("hello world")
```

## Model Catalog

### OpenAI / TikToken

| Model | LLM Compatibility | Vocab | Pre-tokenization | Special Tokens |
|---|---|---|---|---|
| `cl100k_base` | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 | 100,256 | GPT-2 regex | `<\|endoftext\|>`, FIM tokens |
| `o200k_base` | GPT-4o, GPT-4o-mini, GPT-5 | 199,998 | GPT-2 regex | `<\|endoftext\|>`, FIM tokens |
| `p50k_base` | GPT-3 (davinci, curie, babbage, ada) | 50,280 | GPT-2 regex | `<\|endoftext\|>` |
| `r50k_base` | GPT-2 | 50,256 | GPT-2 regex | `<\|endoftext\|>` |

### HuggingFace ByteLevel BPE

| Model | LLM Compatibility | Vocab | Pre-tokenization | Source |
|---|---|---|---|---|
| `qwen25` | Qwen 2.5 (0.5B-72B) | 151,643 | GPT-2 regex | `Qwen/Qwen2.5-0.5B` |
| `phi2` | Microsoft Phi-2 | 50,257 | GPT-2 regex | `microsoft/phi-2` |
| `deepseek-llm` | DeepSeek V2 (7B-Chat) | 100,013 | None (raw) | `deepseek-ai/deepseek-llm-7b-chat` |

## Model Format

All files use the `.tbm` (TinyBPE Model v1) text format:

```
TinyBPE Model v1
<remap_flag>        # 0 = no byte remap, 256 = has remap
[256 remap lines if flag=256]
<left> <right>      # merge pairs, one per line
```

See [`docs/file-formats.md`](../docs/file-formats.md) for the full specification.

## Adding New Models

Convert external tokenizers using the scripts in `scripts/`:

```bash
# TikToken encodings
python scripts/convert_tiktoken.py o200k_base -o models/o200k_base.tbm

# HuggingFace tokenizer.json (local or Hub ID)
python scripts/convert_hf_tokenizer.py Qwen/Qwen2.5-0.5B -o models/qwen25.tbm
```

After conversion, add the model to the registry in `tinybpe/_registry.py` so it becomes available via `Tokenizer.from_pretrained()`.

## Supported LLM Tokenizers

| LLM Family | Tokenizer Type | Status |
|---|---|---|
| GPT-4 / GPT-3.5 | `cl100k_base` BPE | Full support |
| GPT-4o / GPT-5 | `o200k_base` BPE | Full support |
| GPT-3 | `p50k_base` BPE | Full support |
| GPT-2 | `r50k_base` BPE | Full support |
| Qwen 2.5 / 3 | ByteLevel BPE | Full support |
| Phi-2 | ByteLevel BPE | Full support |
| Llama 3 / 4 | ByteLevel BPE | Via `convert_hf_tokenizer.py` |
| DeepSeek V2 / V3 | ByteLevel BPE | Full support |
| MiniCPM5-1B | ByteLevel BPE | Full support |
| Mistral | SentencePiece BPE / Tekken | Not supported |
| Gemma 2 / 3 | SentencePiece Unigram | Not supported |
| Claude 3 / 4 | Proprietary BPE | No public tokenizer |
| Gemini | Proprietary | No public tokenizer |
