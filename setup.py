import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pytorch_lightning_helpers",  # Replace with your own username
    version="1.0",
    author="flyingmilktea",
    author_email="dev@flyingmilktea.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "loguru",
        "numpy",
        "munch",
        "pytorch_lightning",
        "argparse",
        "wandb",
        "hyperpyyaml",
        "nonechucks@git+https://github.com/flyingmilktea/nonechucks.git@main",
        "rich",
        "toolz",
    ],
    extras_require={"dev": ["black", "isort", "yq", "autoflake"]},
)
