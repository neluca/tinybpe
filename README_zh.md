[English](README.md) | [中文]

# 🚀 TinyBPE

[![build](https://github.com/neluca/tinybpe/workflows/build/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/python-package.yml)
[![wheels](https://github.com/neluca/tinybpe/workflows/wheels/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/wheels.yml)
[![lint](https://github.com/neluca/tinybpe/workflows/lint/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/lint.yml)
[![PyPI version](https://img.shields.io/pypi/v/tinybpe)](https://pypi.org/project/tinybpe/)
[![Python versions](https://img.shields.io/pypi/pyversions/tinybpe)](https://pypi.org/project/tinybpe/)
[![License](https://img.shields.io/github/license/neluca/tinybpe)](https://github.com/neluca/tinybpe/blob/main/LICENSE)

**TinyBPE** 是一个高性能、轻量、整洁的语言模型分词器和基本的 **BPE** 模型训练器的 **CPython** 扩展。

## 📦 安装

```bash
pip install tinybpe
```

提供 Linux (x86_64, aarch64)、macOS (x86_64, arm64)、Windows (x86_64) 的预编译 wheel，支持 Python 3.9–3.13。

## 🌟 特性

- 核心由 **C** 语言精心实现，使用 **AVL-Tree** 作为索引，快速高效。
- 以 **Python** 模块形式使用，API 简洁优雅。
- 支持 **BPE** 模型训练和导入模型后继续训练以扩充词表。
- 实现通用的**字节级**分词器，支持快速编解码和**流式解码**。
- 支持正则表达式预分词和添加特殊 **Token**。
- 支持转换 [tiktoken](https://github.com/openai/tiktoken) 的模型参数。
- 核心零依赖，高可定制，易集成和扩展。

## ⚡️ 快速开始

### 📍1. 基本例子

转换 **tiktoken** 的模型参数，创建 **tinybpe** 分词器，并对比 **tiktoken**。

```python
import tiktoken
from tinybpe import Tokenizer, get_from_tiktoken

tik_tokenizer = tiktoken.get_encoding("cl100k_base")
model_param = get_from_tiktoken(tik_tokenizer._mergeable_ranks)  # 转换模型参数
tiny_tokenizer = Tokenizer(model_param)  # 创建 tinybpe 分词器

text = "👋 Hello, this is an example. 你好，这是一个例子。😁"
tik_ids = tik_tokenizer.encode(text)
tiny_ids = tiny_tokenizer.encode(text)
assert tik_ids == tiny_ids  # 输出一致
```

### 📍2. 训练 BPE 模型

```python
from tinybpe import SimpleTrainer

text = open("corpus.txt", "r", encoding="utf-8").read()  # 导入文本文件
trainer = SimpleTrainer(text)  # 创建训练器
vocab_size = 1000  # 词表大小
for _ in range(vocab_size - 256):
    pair, rank, freq = trainer.step()  # 训练
    print(f"{pair} -> {rank} ({freq})")

print(f"词表大小: {trainer.n_merges + 256}")
trainer.save("my-model")  # 保存模型文件 my-model.tinymodel
```

**注意：**
- 模型词表大小 = 256 + 参数大小 (n_merges)。
- 建议对文本做正则预处理，可参考 [examples/regex_trainer.py](examples/regex_trainer.py)。
- 加载已有模型后可继续训练扩充词表，可参考 [examples/simple_continue_training.py](examples/simple_continue_training.py)。

### 📍3. 加载模型并创建分词器

```python
from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("my-model.tinymodel")  # 导入模型文件
tokenizer = Tokenizer(model)  # 创建分词器实例

ids = tokenizer.encode("hello world")
print(ids)                      # [259, 32, 261, 263, 264]
print(tokenizer.decode(ids))    # hello world
print(tokenizer.n_vocab)        # 词表大小
```

### 📍4. 流式解码

```python
def cb_print(text):
    print(text, end="")

decode = tokenizer.stream_decode(cb_print)
for token_id in ids:
    decode(token_id)  # 逐 token 解码并实时打印
```

### 📍5. 转换 tiktoken 模型

```python
import tiktoken
from tinybpe import save_from_tiktoken

enc = tiktoken.get_encoding("cl100k_base")
save_from_tiktoken("cl100k_base", enc._mergeable_ranks)
# 生成 cl100k_base.tinymodel
```

**注意：在商用场景，转换其它分词器的模型需要注意版权问题，建议训练自己的分词器模型。😛**

## 🧪 开发

```bash
git clone https://github.com/neluca/tinybpe.git
cd tinybpe
pip install -r requirements_dev.txt
pip install -e .
python -m pytest
```

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 🤝 感谢

1. 非常感谢 [minbpe](https://github.com/karpathy/minbpe) 对 BPE 算法原理的详细解读和代码实现。
2. 非常感谢 [tiktoken](https://github.com/openai/tiktoken) 提供的分词器模型以供验证。

## 📄 许可证

MIT License。详见 [LICENSE](LICENSE)。
