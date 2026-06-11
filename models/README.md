# TinyBPE Models

Pre-built BPE tokenizer models for popular LLMs.

## Available Models

| Model | Compatible LLM | Vocab Size |
|---|---|---|
| `cl100k_base.tbm` | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 | 100,256 |
| `o200k_base.tbm` | GPT-4o, GPT-4o-mini | 200,256 |
| `p50k_base.tbm` | GPT-3 (davinci, curie, babbage, ada) | 50,257 |
| `r50k_base.tbm` | GPT-2 | 50,257 |

## Usage

```python
from tinybpe import Tokenizer

# Load a pre-built model
tok = Tokenizer.from_file("models/cl100k_base.tbm")

# Encode / decode
ids = tok.encode("Hello world")
text = tok.decode(ids)
```

## Building Models

Models are generated from tiktoken using the conversion script:

```bash
python scripts/convert_tiktoken.py cl100k_base -o models/cl100k_base.tbm
```

See `scripts/README.md` for more conversion options.
