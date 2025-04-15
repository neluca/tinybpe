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
        extra_compile_args={"win32": []}.get(sys.platform, ["-std=c99"]),
    )
]

setup(
    name="tinybpe",
    packages=["tinybpe"],
    ext_modules=ext_modules
)
