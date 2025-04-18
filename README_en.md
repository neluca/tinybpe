[[中文](https://github.com/neluca/tinybpe/blob/main/README.md)|[English](https://github.com/neluca/tinybpe/blob/main/README_en.md)]

# 🚀tinybpe

[![build](https://github.com/neluca/tinybpe/workflows/build/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/python-package.yml)
[![wheels](https://github.com/neluca/tinybpe/workflows/wheels/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/wheels.yml)
[![lint](https://github.com/neluca/tinybpe/workflows/lint/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/lint.yml)
[![coverage](https://img.shields.io/codecov/c/github/neluca/tinybpe)](https://codecov.io/gh/neluca/tinybpe)
[![support-version](https://img.shields.io/pypi/pyversions/tinybpe)](https://pypi.org/project/tinybpe/)
[![license](https://img.shields.io/github/license/neluca/tinybpe)](https://github.com/neluca/tinybpe/blob/main/LICENSE)

👋 `TinyBPE` is a fast, lightweight, and clean **language model** tokenizer and basic **BPE** model trainer, implemented as a **CPython** extension.

## 📦 Setup

```bash
pip install tinybpe
```

## 🌟 Features

- The core is meticulously designed and implemented in **C** , using an **AVL-Tree** as the index for fast and efficient performance.
- Used as a Python module with a simple and elegant `API`.
- Supports training **BPE** models and continuing training on imported models to expand the vocabulary.
- Implements a general **byte-level** tokenizer, supporting fast encoding and decoding,as well asstreaming decoding.
- Supports regular expression pre-tokenization and adding special **Tokens**.
- Supports converting model parameters from [tiktoken](https://github.com/openai/tiktoken).
- Highly customizable, easy to integrate and extend, and the core is zero dependencies.



## ⚡️ Getting Started

### 📍1. Basic Usage

Convert the model parameters of **tiktoken**, create a **tinybpe** tokenizer, and compare it with **tiktoken**.

```python
import tiktoken
from tinybpe import Tokenizer, get_from_tiktoken

tik_tokenizer = tiktoken.get_encoding("cl100k_base")
model_param = get_from_tiktoken(tik_tokenizer._mergeable_ranks)   # Convert model parameters
tiny_tokenizer = Tokenizer(model_param)  # Create a TinyBPE tokenizer

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

### 📍 2. Training a BPE Model

The following code trains a simple **BPE** model. It imports text data from the`<your-text-file>`file, performs no preprocessing on the data, and directly hands it over to `SimpleTrainer`. It then executes the `step()` method `744` times to train a tokenizer with a vocabulary size of `1000`.

```python
from tinybpe import SimpleTrainer

text = open("<your-text-file>", "r", encoding="utf-8").read()  # Import text file
trainer = SimpleTrainer(text)  # Create a trainer
vocab_size = 1000  # Vocabulary size
merges_size = vocab_size - 256  # Model parameter size
for _ in range(merges_size):
    pair, rank, freq = trainer.step()  # Train
    print(f"{pair} -> {rank} ({freq})")  # Print training logs

print(trainer.merges)  # Model parameters
print(trainer.merges_size)  # Model parameter size, which is 744 (1000 - 256)
trainer.save("simple")  # Save the model file as simple.tinymodel
```

**Notes:**

- The model's **vocabulary** size = **256** + the model's **parameter** size (`merges_size`).
- No preprocessing is done on the data. For example, if you directly train on the text string "...hel**lo w**orld..." without preprocessing, there is a chance that the model's vocabulary will contain tokens like "lo w". Therefore, it is recommended to preprocess the text file. You can refer to [examples/regex_trainer.py](https://github.com/neluca/tinybpe/blob/main/examples/regex_trainer.py) .
- After loading the model parameters, you can continue training on the data to expand the vocabulary. You can refer to [examples/simple_continue_training.py](https://github.com/neluca/tinybpe/blob/main/examples/simple_continue_training.py) .

For more complex **BPE** models, you can design your own data preprocessing functions or inherit from the parent class `bpe.Trainer` of `SimpleTrainer` to implement your own trainer. `bpe.Trainer` is a high-performance, highly customizable base trainer implemented in **C**. You can also load an existing **tinybpe** model and continue training on text data to expand your vocabulary.

### 📍3. Loading a Model and Creating a Tokenizer

Use `load_bpe_model` to import the model file as tokenizer parameters, and then create a `Tokenizer` instance from the model parameters.

```python
from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("simple.tinymodel")  # Import the model file
tokenizer = Tokenizer(model)  # Create a tokenizer instance
s1 = "hello world, old man !"
ids = tokenizer.encode(s1)  # Encode
print(ids)
s2 = tokenizer.decode(ids)  # Decode
print(s2)
print(tokenizer.n_vocab)  # Output the vocabulary size
tokenizer.save_vocab("simple")  # Export the vocabulary file as simple.vocab
```

The `Tokenizer` has three parameters. The other two parameters are `pat_str` and `special_tokens` , which serve the same purpose as in **tiktoken**. `pat_str` is a regular expression string that preprocesses the text string for the tokenizer, and `special_tokens` is a dictionary of special **Tokens**. For more details, you can refer to [examples/regex_tokenizer.py](https://github.com/neluca/tinybpe/blob/main/examples/regex_tokenizer.py) and [examples/cl100k_tokenizer.py](https://github.com/neluca/tinybpe/blob/main/examples/cl100k_tokenizer.py).

### 📍4. Streaming Decoding

Streaming decoding means decoding a string one **Token ID** at a time. If there are insufficient bytes to decode using unicode,the program will internally cache these bytes until they can be properly decoded.

```python
from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("simple.tinymodel")
tokenizer = Tokenizer(model)

s = "hello world 你好世界 😁"
ids = tokenizer.encode(s)
# String processing function
def cb_print(text):
    print(text, end="")

decode = tokenizer.stream_decode(cb_print)  # Generate a streaming decoding function
for i in ids:
    decode(i)  # Decode one Token ID at a time
```

### 📍5. Converting a Tiktoken Model

Convert the model parameters of tiktoken, specifically the `mergeable_ranks` , into a model file that can be loaded by **tinybpe**.

```python
import tiktoken
from tinybpe import save_from_tiktoken

enc = tiktoken.get_encoding("cl100k_base")
save_from_tiktoken("cl100k_base", enc._mergeable_ranks)  # Save the tiktoken model parameters as a TinyBPE model file
```

After running the above code, a file named ***cl100k_base.tinymodel*** will appear in the directory. You can load this model as shown in step 3 and use it normally, for example: [examples/cl100k_tokenizer.py](https://github.com/neluca/tinybpe/blob/main/examples/cl100k_tokenizer.py).

**Note: In commercial scenarios, converting models from other tokenizers may involve copyright issues. Therefore, it is recommended to train your own tokenizer model. 😛**



## 🔧 Contributing

Welcome contributions! If you find any **bugs** or have suggestions and ideas for improvement, feel free to open an **issue** to discuss them. If you want to add your creative ideas to the code or fix a **bug**, feel free to submit a pull request.

## 🤝 Acknowledgments

1. Very grateful to [minbpe](https://github.com/karpathy/minbpe) for its detailed explanation of the BPE algorithm and the corresponding code implementation.

2. Very grateful to [tiktoken](https://github.com/openai/tiktoken) for providing tokenizer models for validation.


## ⌛Unit Testing


```bash
pip install -r requirements_dev.txt
python build_setup.py build_ext
python -m pytest
```

