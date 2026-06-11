# TinyBPE Conversion Scripts

Scripts for converting various tokenizer model formats to TinyBPE `.tbm` files.

## Scripts

### `convert_tiktoken.py`

Convert OpenAI [tiktoken](https://github.com/openai/tiktoken) encodings to TinyBPE.

```bash
# Install tiktoken first
pip install tiktoken

# Convert a built-in encoding
python scripts/convert_tiktoken.py cl100k_base -o models/cl100k_base.tbm
python scripts/convert_tiktoken.py o200k_base -o models/o200k_base.tbm
python scripts/convert_tiktoken.py p50k_base -o models/p50k_base.tbm
python scripts/convert_tiktoken.py r50k_base -o models/r50k_base.tbm
```

### `convert_hf_tokenizer.py`

Convert HuggingFace `tokenizer.json` files to TinyBPE.

```bash
# Install huggingface_hub first
pip install huggingface_hub

# Convert from a local tokenizer.json
python scripts/convert_hf_tokenizer.py path/to/tokenizer.json -o models/my_model.tbm

# Convert from a HuggingFace model ID
python scripts/convert_hf_tokenizer.py meta-llama/Meta-Llama-3-8B -o models/llama3.tbm
```

## Adding New Scripts

When adding a new conversion script:

1. Follow the existing pattern: CLI with `argparse`, `-o` for output path
2. Output TinyBPE `.tbm` models via `tinybpe._model_io.save_model()`
3. Add documentation to this README
4. Do NOT add conversion code to the `tinybpe` package itself — keep it in `scripts/`
