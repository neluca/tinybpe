# 🚀tinybpe

👋 **tinybpe** is a fast, lightweight, and clean **language model** tokenizer and **BPE** trainer.

## 📦 Setup

```bash
python -m build
```

## 🌟 Features

- Core is thoughtfully designed and implemented in **C** language, fast and efficient.
- Used as a **Python** module, simple and elegant.
- Supports **BPE** model training and importing models for continued training to expand the vocabulary.
- Implements a general byte-level tokenizer, supporting encoding, decoding, and streaming decoding.
- Supports regular expression pre-tokenization and adding special **Tokens**.
- Supports conversion of **tiktoken** model parameters.
- Very easy to integrate and extend, and the core is zero dependencies.



## ⚡️ Getting Started

#### 📍 1. Train a BPE Mode

The following code trains a simple **BPE** model. It imports text data without any preprocessing and directly uses `SimpleTrainer` to train a tokenizer with a vocabulary size of `1000` .

```python
from tinybpe import SimpleTrainer

text = open("the-old-man-and-the-sea.txt", "r", encoding="utf-8").read()  # Import text file
trainer = SimpleTrainer(text)  # Create trainer
vocab_size = 1000  # Vocabulary size
merges_size = vocab_size - 256  # Model parameter size
for _ in range(merges_size):
    pair, rank, freq = trainer.step()  # Train
    print(f"{pair} -> {rank} ({freq})")  # Print training logs

print(trainer.merges)  # Model parameters
print(trainer.merges_size)  # Model parameter size (here it is 744, i.e., 1000 - 256)
trainer.save("simple")  # Save model file as simple.tinymodel
```

**Note**: The model's **vocabulary size** = **256** + model's **parameter size** (`merges_size`).

To train a more complex **BPE** model, you can design your own data preprocessing functions or inherit from `SimpleTrainer`. You can also load an existing **tinybpe** model and continue training to expand the vocabulary. For more details, refer to the examples in the 📂**examples** folder.

