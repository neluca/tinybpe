"""Minimal setup.py — only defines the C extension.

All package metadata lives in pyproject.toml.
"""

import sys
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "tinybpe.bpe",
        sources=[
            "src/bpe_module.c",
            "src/_tree_core.c",
            "src/bpe_common.c",
            "src/bpe_trainer.c",
            "src/bpe_tokenizer.c",
        ],
        depends=[
            "src/_tree_core.h",
            "src/bpe_common.h",
            "src/bpe_trainer.h",
            "src/bpe_tokenizer.h",
        ],
        # NB: on 64-bit Windows, sys.platform is "win32" (historical).
        # MSVC uses /W* flags instead of -W*, so pass nothing here.
        extra_compile_args={
            "win32": [],
        }.get(sys.platform, ["-Wall", "-Wextra", "-std=c99"]),
    )
]

setup(
    ext_modules=ext_modules,
)
