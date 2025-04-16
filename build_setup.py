import sys, os, shutil
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "bpe",
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
        extra_compile_args={"win32": []}.get(sys.platform, ["-Werror", "-std=c99"]),
    )
]

setup(
    name="tinybpe",
    ext_modules=ext_modules
)

bpe_file_path = None

for root, dirs, files in os.walk("build"):
    for file in files:
        if (file.endswith(".pyd") or file.endswith(".so") or file.endswith(".dylib")) \
                and file.startswith("bpe"):
            bpe_file_path = os.path.join(root, file)
            break

if bpe_file_path is not None:
    shutil.copy(bpe_file_path, "tinybpe")
else:
    raise RuntimeError("Error ...")
