# TinyBPE Models

Pre-built BPE tokenizer models for popular LLMs, converted to TinyBPE `.tbm` format.

## Available Models

| Model File | LLM Compatibility | Vocab Size | Byte Remap |
|---|---|---|---|
| `cl100k_base.tbm` | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 | 100,256 | Yes |
| `o200k_base.tbm` | GPT-4o, GPT-4o-mini, GPT-5 | 199,998 | Yes |
| `p50k_base.tbm` | GPT-3 (davinci, curie, babbage, ada) | 50,280 | Yes |
| `r50k_base.tbm` | GPT-2 | 50,256 | Yes |

## Model Sources

All models are converted from [OpenAI tiktoken](https://github.com/openai/tiktoken) encodings:

| Encoding | Description |
|---|---|
| `cl100k_base` | 100K vocab, best for GPT-4 / GPT-3.5. English-optimized. |
| `o200k_base` | 200K vocab, best for GPT-4o / GPT-5. Multilingual-friendly. |
| `p50k_base` | 50K vocab, used by GPT-3 davinci models. |
| `r50k_base` | 50K vocab, used by GPT-2. |

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

| LLM Family | Tokenizer Type | Conversion Method |
|---|---|---|
| **Llama 3/4** | tiktoken-style BPE (128K) | `scripts/convert_hf_tokenizer.py` from `tokenizer.json` |
| **Qwen 2.5/3** | BPE (151.9K) | `scripts/convert_hf_tokenizer.py` from `tokenizer.json` |
| **DeepSeek V2/V3** | BPE (~128K) | `scripts/convert_hf_tokenizer.py` from `tokenizer.json` |
| **Mistral** | Tekken BPE / SentencePiece | `scripts/convert_hf_tokenizer.py` from `tokenizer.json` |
| **Gemma 2/3** | SentencePiece Unigram (256K) | Not compatible (different algorithm) |
| **Claude 3/4** | Proprietary BPE | Not available (no public tokenizer) |
| **Gemini** | Proprietary SentencePiece | Not available (use Vertex AI SDK) |

For HuggingFace-based tokenizers, see `scripts/README.md` for conversion instructions.
