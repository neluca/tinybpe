# TinyBPE Benchmarks

Performance benchmarks for the tinybpe CPython extension.

## Running Benchmarks

```bash
cd benchmarks
python bench_encode.py
python bench_decode.py
python bench_train.py
```

## Benchmarks

- **bench_encode.py**: Measures BPE encoding throughput (bytes → token IDs) across input sizes from 1KB to 1MB.
- **bench_decode.py**: Measures BPE decoding throughput (token IDs → bytes) across token list sizes from 100 to 100,000.
- **bench_train.py**: Measures BPE training speed across training corpus sizes from 10KB to 1MB.

## Interpreting Results

- **Encode throughput** (MB/s): How fast text can be tokenized. Higher is better.
- **Decode throughput** (tokens/ms): How many tokens can be decoded per millisecond. Higher is better.
- **Training speed** (ms/step): Time per BPE merge step. Lower is better.

For comparison purposes, you can run equivalent benchmarks against tiktoken:
```bash
pip install tiktoken
python -c "
import tiktoken, time
enc = tiktoken.get_encoding('cl100k_base')
text = b'hello world ' * 10000
t0 = time.perf_counter()
for _ in range(100):
    enc.encode(text.decode())
print(f'tiktoken: {(time.perf_counter() - t0) / 100 * 1000:.2f} ms')
"
```
