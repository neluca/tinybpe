# TinyBPE

[![PyPI version](https://img.shields.io/pypi/v/tinybpe)](https://pypi.org/project/tinybpe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tinybpe/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**纯 C 内核的超轻量高性能 BPE 分词器与训练器。**

只需**一行代码**即可加载 GPT-4 兼容分词器，无需联网。TinyBPE 内置 8 个 ByteLevel BPE 预训练模型，开箱即用。CPython C 扩展以原生速度执行 BPE 编码/解码——通常比纯 Python 实现**快 10–50 倍**，且仅依赖 `regex` 一个第三方库。

## 为什么选择 TinyBPE？

| 特性 | TinyBPE | tiktoken | HuggingFace tokenizers |
|---|---|---|---|
| **核心引擎** | 纯 C（CPython 扩展） | 纯 Rust（PyO3） | 纯 Rust（PyO3） |
| **依赖** | 仅 `regex` | `tiktoken` + Rust 工具链 | `tokenizers` + Rust 工具链 |
| **内置模型** | 8 个模型随包分发 | 首次使用时下载 | 首次使用时下载 |
| **离线可用** | ✅ 完全离线 | ❌ 需要下载 | ❌ 需要下载 |
| **模型格式** | 可读文本 `.tbm` 文件 | 二进制 blob | JSON / 二进制 |
| **一行加载** | `Tokenizer.from_pretrained("cl100k_base")` | `tiktoken.get_encoding("cl100k_base")` | `AutoTokenizer.from_pretrained(...)` |
| **训练新模型** | ✅ 纯 C 训练器 | ❌ | ✅（需要 Rust 编译） |
| **流式解码** | ✅ UTF-8 边界缓存 | ❌ | ❌ |
| **可移植 C 内核** | ✅ 可嵌入式部署 | ❌ | ❌ |
| **安装体积** | 约 3 MB（压缩后） | 约 2 MB + 缓存模型 | 约 4 MB + 缓存模型 |

## 安装

```bash
pip install tinybpe
```

可选扩展：

```bash
pip install tinybpe[dev]       # 开发工具（pytest、ruff、mypy）
pip install tinybpe[tiktoken]  # 用于与 tiktoken 对比测试
pip install tinybpe[hf]        # 用于 HuggingFace 模型转换
pip install tinybpe[all]       # 安装全部可选依赖
```

## 快速开始

### 一行代码加载模型

```python
from tinybpe import Tokenizer

# 加载任意内置模型——无需联网，无需下载
tok = Tokenizer.from_pretrained("cl100k_base")

ids = tok.encode("hello world")
tok.decode(ids)  # → 'hello world'
```

### 查看可用模型

```python
import tinybpe

tinybpe.list_models()
# ['cl100k_base', 'deepseek-v4', 'minicpm', 'o200k_base',
#  'p50k_base', 'phi2', 'qwen35', 'r50k_base']
```

### 内置模型目录

| 模型名称 | 兼容模型 | 词表大小 |
|---|---|---|
| `cl100k_base` | GPT-4、GPT-3.5-turbo、text-embedding-ada-002 | 100,256 |
| `o200k_base` | GPT-4o、GPT-4o-mini、GPT-5 | 199,998 |
| `p50k_base` | GPT-3（davinci、curie、babbage、ada） | 50,280 |
| `r50k_base` | GPT-2 | 50,256 |
| `qwen35` | Qwen3.5（0.8B-35B） | 247,843 |
| `phi2` | Microsoft Phi-2 | 50,257 |
| `deepseek-v4` | DeepSeek-V4 Flash | 127,997 |
| `minicpm5` | MiniCPM5-1B（ByteLevel BPE） | 130,050 |

### 训练分词器

```python
from tinybpe import Trainer

trainer = Trainer("hello world " * 500)
trainer.train(100)          # 学习 100 个合并规则
trainer.save("my_model")    # → my_model.tbm
```

### 流式解码

```python
parts = []
decoder = tok.stream_decode(lambda s: parts.append(s))
for tid in ids:
    decoder(tid)
assert "".join(parts) == "hello world"
```

### 正则预分词

```python
PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

tok = Tokenizer.from_file("my_model.tbm", pat_str=PAT)
```

### 特殊 Token

```python
special_tokens = {"<eot>": 1000, "<fim_prefix>": 1001, "<fim_suffix>": 1002}
tok = Tokenizer(merges, special_tokens=special_tokens)
ids = tok.encode("<fim_prefix> hello world <eot>")
```

### 字节重映射（兼容 TikToken）

```python
from tinybpe import load_model

merges, bytes_maps = load_model("cl100k_base.tbm")
tok = Tokenizer(merges, bytes_maps=bytes_maps)
```

## API 参考

### `Tokenizer`

```python
class Tokenizer:
    def __init__(self, merges, *, bytes_maps=None, pat_str=None, special_tokens=None)
    def encode(self, text: str) -> list[int]
    def encode_ordinary(self, text: str) -> list[int]
    def decode(self, ids: list[int]) -> str
    def stream_decode(self, callback: Callable[[str], None]) -> Callable[[int], None]
    def stream_decode_reset(self) -> None
    def save(self, path: str) -> None
    def save_vocab(self, path: str) -> None

    @classmethod
    def from_file(cls, path: str, *, pat_str=None, special_tokens=None) -> Tokenizer
    @classmethod
    def from_pretrained(cls, name: str) -> Tokenizer

    @property
    def merges(self) -> list[tuple[int, int]]
    @property
    def vocab(self) -> dict[int, bytes]
    @property
    def n_vocab(self) -> int
```

### `Trainer`

```python
class Trainer(bpe.Trainer):
    def __init__(self, text, *, preprocess=None, callback=None)
    def step(self) -> tuple | None
    def train(self, n: int) -> int
    def save(self, path: str) -> None

    @property
    def merges(self) -> list[tuple[int, int]]
    @property
    def n_merges(self) -> int
```

### 模型发现

```python
def list_models() -> list[str]
```

### 文件 I/O

```python
def load_model(path: str) -> tuple[list[tuple[int, int]], list[int] | None]
def save_model(path: str, merges, bytes_maps=None) -> None
def load_vocab(path: str) -> dict[int, bytes]
def save_vocab(path: str, vocab: dict[int, bytes]) -> None
```

## 模型格式

`.tbm`（TinyBPE Model）是可读文本文件：

```
TinyBPE Model v1
0               # 0 = 无字节重映射，256 = 有字节重映射
104 101         # 合并规则，每行一对
256 108
...
```

详见 [`docs/file-formats.md`](docs/file-formats.md)。

## 转换脚本

将其他分词器格式转换为 TinyBPE 格式：

```bash
# TikToken
python scripts/convert_tiktoken.py cl100k_base -o models/cl100k_base.tbm

# HuggingFace
python scripts/convert_hf_tokenizer.py tokenizer.json -o output.tbm
python scripts/convert_hf_tokenizer.py Qwen/Qwen3.5-0.8B -o models/qwen35.tbm

```

详见 [`scripts/README.md`](scripts/README.md)。

## 性能

C 内核使用 AVL 树实现 O(log n) 的合并对查找，编码时采用贪心最低秩优先合并策略。在现代 CPU 上的典型吞吐量：

| 操作 | 吞吐量 |
|---|---|
| 训练（C 内核） | 约 500–1000 万字符/秒 |
| 编码（C 内核） | 约 200–500 万 token/秒 |
| 解码（C 内核） | 约 1000–2000 万 token/秒 |

运行本地基准测试：

```bash
python benchmarks/bench_train.py
python benchmarks/bench_encode.py
python benchmarks/bench_decode.py
```

## 项目优势

- **零网络依赖** —— 所有模型随包分发，离线可用
- **极简依赖** —— 仅需 `regex`，无需 Rust 工具链
- **纯 C 高性能** —— C 扩展内核，编译为本地机器码
- **可嵌入部署** —— C 训练器和分词器可脱离 Python 独立使用
- **可读模型文件** —— `.tbm` 纯文本格式，可直接检视和调试
- **生产就绪** —— 95%+ 测试覆盖率、严格 mypy 类型检查、跨平台 CI/CD

## 开发

```bash
git clone https://github.com/neluca/tinybpe.git
cd tinybpe
pip install -e ".[dev]"
make test && make lint && make typecheck
```

详见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
