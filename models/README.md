# TinyBPE Models

Pre-built BPE tokenizer models for popular LLMs, converted to TinyBPE `.tbm` format.

## Available Models

### TikToken (OpenAI)

| Model File | LLM Compatibility | Vocab Size | Source |
|---|---|---|---|
| `cl100k_base.tbm` | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 | 100,256 | tiktoken |
| `o200k_base.tbm` | GPT-4o, GPT-4o-mini, GPT-5 | 199,998 | tiktoken |
| `p50k_base.tbm` | GPT-3 (davinci, curie, babbage, ada) | 50,280 | tiktoken |
| `r50k_base.tbm` | GPT-2 | 50,256 | tiktoken |

### HuggingFace (ByteLevel BPE)

| Model File | LLM Compatibility | Vocab Size | Source |
|---|---|---|---|
| `qwen25.tbm` | Qwen 2.5 (0.5B–72B) | 151,643 | `Qwen/Qwen2.5-0.5B` |
| `phi2.tbm` | Phi-2 | 50,257 | `microsoft/phi-2` |
| `deepseek-llm.tbm` | DeepSeek V2 (7B-Chat) | 100,013 | `deepseek-ai/deepseek-llm-7b-chat` |

## Usage

```python
from tinybpe import Tokenizer

# Load a pre-built model
tok = Tokenizer.from_file("models/cl100k_base.tbm")

# Encode / decode
ids = tok.encode("Hello world")
text = tok.decode(ids)
```

## Model Conversion

Models are generated using the conversion scripts:

```bash
# Convert all tiktoken encodings
python scripts/convert_tiktoken.py cl100k_base -o models/cl100k_base.tbm
python scripts/convert_tiktoken.py o200k_base -o models/o200k_base.tbm
python scripts/convert_tiktoken.py p50k_base -o models/p50k_base.tbm
python scripts/convert_tiktoken.py r50k_base -o models/r50k_base.tbm
```

## Other Mainstream LLM Tokenizers

| LLM Family | Tokenizer Type | `.tbm` Support |
|---|---|---|
| **GPT-4 / GPT-3.5** | tiktoken `cl100k_base` BPE | ✅ Full |
| **GPT-4o / GPT-5** | tiktoken `o200k_base` BPE | ✅ Full |
| **GPT-3** | tiktoken `p50k_base` BPE | ✅ Full |
| **GPT-2** | tiktoken `r50k_base` BPE | ✅ Full |
| **Qwen 2.5/3** | ByteLevel BPE (151.9K) | ✅ Full |
| **Phi-2** | ByteLevel BPE (50K) | ✅ Full |
| **Llama 3/4** | ByteLevel BPE (128K) | ✅ Via `convert_hf_tokenizer.py`¹ |
| **DeepSeek V2/V3** | ByteLevel BPE (~100K) | ✅ Full³ |
| **Mistral** | SentencePiece BPE / Tekken | ❌ Different format³ |
| **Gemma 2/3** | SentencePiece Unigram (256K) | ❌ Different algorithm |
| **Claude 3/4** | Proprietary BPE | ❌ No public tokenizer |
| **Gemini** | Proprietary | ❌ No public tokenizer |

¹ Llama requires authentication to download the tokenizer from HuggingFace.
² SentencePiece uses Metaspace pre-tokenization, not ByteLevel.
³ DeepSeek omits 13 single-char tokens for invalid UTF-8 bytes (0xC0-0xC1, 0xF5-0xFF). Converter assigns unused IDs in 0-255 range to maintain bijective mapping.

See `scripts/README.md` for conversion instructions.
