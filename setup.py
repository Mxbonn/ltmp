from setuptools import find_packages, setup

setup(
    name="ltmp",
    version="0.1",
    author="Maxim Bonnaerens",
    description="Learned Thresholds Token Merging and Pruning for Vision Transformers",
    install_requires=[
        "torchvision",
        "numpy",
        "timm @ git+ssh://git@github.com/rwightman/pytorch-image-models.git@8ff45e41f7a6aba4d5fdadee7dc3b7f2733df045",
        "pillow",
        "tqdm",
        "scipy",
        "fvcore",
    ],
    packages=find_packages(exclude=("build")),
)
