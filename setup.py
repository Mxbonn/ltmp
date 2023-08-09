from setuptools import find_packages, setup

setup(
    name="ltmp",
    version="0.1",
    author="Maxim Bonnaerens",
    description="Learned Thresholds Token Merging and Pruning for Vision Transformers",
    install_requires=[
        "torchvision",
        "numpy",
        "timm==0.9.5",
        "pillow",
        "tqdm",
        "scipy",
        "fvcore",
    ],
    packages=find_packages(exclude=("build")),
)
