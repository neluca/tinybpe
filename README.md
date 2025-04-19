[中文|[English](https://github.com/neluca/tinybpe/blob/main/README_en.md)]

# 🚀tinybpe

[![build](https://github.com/neluca/tinybpe/workflows/build/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/python-package.yml)
[![wheels](https://github.com/neluca/tinybpe/workflows/wheels/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/wheels.yml)
[![lint](https://github.com/neluca/tinybpe/workflows/lint/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/lint.yml)
[![coverage](https://codecov.io/gh/neluca/tinybpe/branch/main/graph/badge.svg)](https://codecov.io/gh/neluca/tinybpe)
[![support-version](https://img.shields.io/pypi/pyversions/tinybpe)](https://pypi.org/project/tinybpe/)
[![license](https://img.shields.io/github/license/neluca/tinybpe)](https://github.com/neluca/tinybpe/blob/main/LICENSE)

👋 **TinyBPE** 是一个包含高性能、轻量、整洁的语言模型分词器和基本的 **BPE** 模型训练器的 **CPython** 扩展。

## 📦安装

```bash
pip install tinybpe
```

## 🌟特性：

- 核心由 **C** 语言精心设计实现，使用 **AVL-Tree** 作为索引，快速高效。
- 以 **Python** 模块的形式使用，`API`简洁优雅。
- 支持 **BPE** 模型训练和导入模型后继续训练以扩充词表。
- 实现通用的字节级分词器，支持快速编解码和<u>流式解码</u>。
- 支持正则表达式预分词和添加特殊 **Token** 。
- 支持转换 [tiktoken](https://github.com/openai/tiktoken) 的模型参数。
- 十分容易集成和扩展，高可定制，核心零依赖。



## ⚡️快速开始

### 📍1、基本例子

转换 **tiktoken** 的模型参数，创建 **tinybpe** 分词器，并对比 **tiktoken**。

```python
import tiktoken
from tinybpe import Tokenizer, get_from_tiktoken

tik_tokenizer = tiktoken.get_encoding("cl100k_base")
model_param = get_from_tiktoken(tik_tokenizer._mergeable_ranks)   # 转换模型参数
tiny_tokenizer = Tokenizer(model_param)  # 创建tinybpe分词器 

text = "👋 Hello, this is an example. 你好，这是一个例子。😁"
tik_ids = tik_tokenizer.encode(text)
tiny_ids = tiny_tokenizer.encode(text)
assert tik_ids == tiny_ids

print("tiktoken: ", tik_ids)
print("tinybpe: ", tiny_ids)

tik_text = tik_tokenizer.decode(tiny_ids)
tiny_text = tiny_tokenizer.decode(tik_ids)
assert tik_text == tiny_text

print("tiktoken: ", tik_text)
print("tinybpe: ", tiny_text)

# output:
# tiktoken:  [9468, 239, 233, 22691, 11, 420, 374, 459, 3187, 13, 220, 57668, 53901, 3922, 44388, 21043, 48044, 27452, 45829, 1811, 76460, 223]
# tinybpe:  [9468, 239, 233, 22691, 11, 420, 374, 459, 3187, 13, 220, 57668, 53901, 3922, 44388, 21043, 48044, 27452, 45829, 1811, 76460, 223]
# tiktoken:  👋 Hello, this is an example. 你好，这是一个例子。😁
# tinybpe:  👋 Hello, this is an example. 你好，这是一个例子。😁
```

### 📍2、训练BPE模型

下列代码训练一个简单的**BPE**模型，导入`<your-text-file>`文件中的文本数据，不对数据做任何预处理，直接交由`SimpleTrainer` ，执行 `setp()` 方法 `744`次，训练出一个词表大小为`1000` 的分词器。

```python
from tinybpe import SimpleTrainer

text = open("<your-text-file>", "r", encoding="utf-8").read()  # 导入文本文件
trainer = SimpleTrainer(text)  # 创建训练器
vocab_size = 1000  # 词表大小
merges_size = vocab_size - 256  # 模型的参数大小
for _ in range(merges_size):
    pair, rank, freq = trainer.step()  # 训练
    print(f"{pair} -> {rank} ({freq})")  # 打印训练时的日志

print(trainer.merges)  # 模型参数
print(trainer.merges_size)  # 模型的参数大小，这里是 744 (1000 - 256)
trainer.save("simple")  # 保存模型文件 simple.tinymodel
```

注意：

- 模型的**词表**大小 = **256** + 模型的**参数**大小(**merges_size**)。
- 不对数据做任何预处理，例如对文本字符串 "... hel**lo w**orld ..."直接训练，模型的词表中有概率会出现"lo w"这样的词，所以建议预处理一下文本文件，可以参考 [examples/regex_trainer.py](https://github.com/neluca/tinybpe/blob/main/examples/regex_trainer.py) 。
- 加载模型参数后，可以对数据做继续训练以扩充词表，可以参考 [examples/simple_continue_training.py](https://github.com/neluca/tinybpe/blob/main/examples/simple_continue_training.py) 。

训练复杂的**BPE**模型，可以按你自己的需求，设计数据预处理函数或者继承`SimpleTrainer`的父类`bpe.Trainer`，实现属于你的训练器，`bpe.Trainer`由 **C** 实现的高性能、高可定制的基础训练器；你也可以加载已有的**tinybpe**模型，在文本数据上继续训练，扩充你自己的词表。

### 📍3、加载模型并创建分词器

通过`load_bpe_model`将模型文件导入为分词器参数，再由模型的参数创建 `Tokenizer` 实例。

```python
from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("simple.tinymodel")  # 导入模型文件
tokenizer = Tokenizer(model)  # 创建分词器实例
s1 = "hello world, old man !"
ids = tokenizer.encode(s1)  # 编码
print(ids)
s2 = tokenizer.decode(ids)  # 解码
print(s2)
print(tokenizer.n_vocab)  # 输出词表大小
tokenizer.save_vocab("simple")  # 导出词表文件 simple.vocab
```

`Tokenizer` 有三个参数，另外两个参数分别是 `pat_str` 和 `special_tokens`，作用和 **tiktoken** 一致。`pat_str` 是一个正则表达式字符串，负责对分词器的文本字符串做预处理；`special_tokens` 为添加的特殊 **Tokens** 字典。详情可以参考 [examples/regex_tokenizer.py](https://github.com/neluca/tinybpe/blob/main/examples/regex_tokenizer.py) 和 [examples/cl100k_tokenizer.py](https://github.com/neluca/tinybpe/blob/main/examples/cl100k_tokenizer.py) 。

### 📍4、流式解码

流式解码，即用一个一个的 **Token ID** 解码出字符串，遇到不足以用 *unicode* 解码的字节，程序内部会选择缓存该字节，直到能够正常解码为止。

```python
from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("simple.tinymodel")
tokenizer = Tokenizer(model)

s = "hello world 你好世界 😁"
ids = tokenizer.encode(s)
# 字符串处理函数
def cb_print(text):
    print(text, end="")

decode = tokenizer.stream_decode(cb_print)  # 生成流式解码的解码函数
for i in ids:
    decode(i)  # 一个一个用 Token ID 去解码
```

### 📍5、转换tiktoken模型

将 **tiktoken** 的模型参数，也就是将 `mergeable_ranks` 保存能够被 **tinybpe** 正常加载的模型文件。

```python
import tiktoken
from tinybpe import save_from_tiktoken

enc = tiktoken.get_encoding("cl100k_base")
save_from_tiktoken("cl100k_base", enc._mergeable_ranks)  # 将 tiktoken 模型参数保存为 tinybpe 的模型文件
```

执行以上代码后，目录中会出现一个名为 ***cl100k_base.tinymodel*** 的文件，只需要像第三步一样去加载这个模型，就可以正常使用了， 例如：[examples/cl100k_tokenizer.py](https://github.com/neluca/tinybpe/blob/main/examples/cl100k_tokenizer.py)。

**注意：在商用场景，转换其它分词器的模型需要注意版权问题，所以建议自己训练属于自己的分词器模型😛。**



## 🔧贡献

欢迎贡献代码，如果您发现了 **bug** 或者有任何建议和改进意见，欢迎开启一个 **issue** 来讨论；如果需要往代码中加入您的创意，或者修复某个 **bug**，欢迎提交 **pull request**。

## 🤝感谢

1. 非常感谢 [minbpe](https://github.com/karpathy/minbpe) 对BPE算法原理的详细解读和相应的代码实现。

2. 非常感谢 [tiktoken](https://github.com/openai/tiktoken) 提供的分词器模型以供验证。


## ⌛单元测试

```bash
pip install -r requirements_dev.txt
python build_setup.py build_ext
python -m pytest
```

