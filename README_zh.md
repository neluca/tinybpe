# 🚀tinybpe

👋 **tinybpe** 是一个高性能、轻量、整洁的**语言模型**分词器和 **BPE** 训练器。



## 📦安装

```bash
pip install tinybpe
```



## 🌟特性：

- 核心由 **C** 语言精心设计实现，快速高效
- 以 **Python** 模块的形式使用，简洁优雅
- 支持 **BPE** 模型训练和导入模型后继续训练以扩充词表
- 实现通用的字节级分词器，支持快速编解码和流式解码
- 支持正则表达式预分词和添加特殊 **Token** 

- 支持转换 **tiktoken** 的模型参数
- 十分容易集成和扩展，核心零依赖



## ⚡️快速开始

#### 📍1、训练BPE模型

下列代码是训练一个简单的**BPE**模型，导入文本数据，不对数据做任何预处理，直接交由`SimpleTrainer` 训练出一个词表大小为`1000` 的分词器。

```python
from tinybpe import SimpleTrainer

text = open("the-old-man-and-the-sea.txt", "r", encoding="utf-8").read()  # 导入文本文件
trainer = SimpleTrainer(text)  # 创建训练器
vocab_size = 1000  # 词表大小
merges_size = vocab_size - 256  # 模型的参数大小
for _ in range(merges_size):
    pair, rank, freq = trainer.step()  # 训练
    print(f"{pair} -> {rank} ({freq})")  # 打印训练时的日志

print(trainer.merges)  # 模型参数，类似与
print(trainer.merges_size)  # 模型的参数大小，这里是 744 (1000 - 256)
trainer.save("simple")  # 保存模型文件 simple.tinymodel
```

注意：模型的**词表**大小 = **256** + 模型的**参数**大小(**merges_size**)

训练复杂的**BPE**模型，可以按你自己的需求，设计数据预处理函数或者继承`SimpleTrainer`，也可以加载已有的可用的**tinybpe**模型，继续训练以扩充词表，详情可以参考📂**examples** 里面的例子。



#### 📍2、加载模型创建分词器

通过`load_bpe_model`将模型文件导入为分词器模型参数，再由模型参数创建 `Tokenizer` 实例。

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

`Tokenizer` 有三个参数，另外两个参数分别是 `pat_str` 和 `special_tokens`，作用和 **tiktoken** 一致。`pat_str` 是一个正则表达式字符串，负责分词器的预分词；`special_tokens` 为添加的特殊 **Tokens** 字典。详情可以参考📂**examples** 里面的例子。



#### 📍3、流式解码

流式解码即用一个一个的 **Token ID** 解码出字符串，遇到不足以用 *unicode* 解码的字节，内部会缓存该字节，直到能够正常解码为止。

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



#### 📍4、转换tiktoken模型

将 **tiktoken** 的模型参数，也就是将 `mergeable_ranks` 保存能够被 **tinybpe** 正常加载的模型文件。

```python
import tiktoken
from tinybpe import save_from_tiktoken

enc = tiktoken.get_encoding("cl100k_base")
save_from_tiktoken("cl100k_base", enc._mergeable_ranks)  # 将 tiktoken 参数保存为 tinybpe 的模型文件
```

详情可以参考📂**examples** 里面使用 **tiktoken** 模型参数的例子。



## ⌛单元测试

```bash
pip install -r requirements_dev.txt
python -m pytest
```

