# Learned Thresholds Token Merging and Pruning for Vision Transformers 
## [[`Transactions on Machine Learning Research (TMLR)`](https://openreview.net/forum?id=WYKTCKpImz)] [[`arXiv`](https://arxiv.org/abs/2307.10780)] [[`Project site`](https://maxim.bonnaerens.com/publication/ltmp)]

By Maxim Bonnaerens, and Joni Dambre.

## Abstract
Vision transformers have demonstrated remarkable success in a wide range of computer vision tasks over the last years. However, their high computational costs remain a significant barrier to their practical deployment.
In particular, the complexity of transformer models is quadratic with respect to the number of input tokens.
Therefore techniques that reduce the number of input tokens that need to be processed have been proposed.
This paper introduces Learned Thresholds token Merging and Pruning (**LTMP**), **a novel approach that leverages the strengths of both token merging and token pruning**.
LTMP uses **learned threshold** masking modules that dynamically determine which tokens to merge and which to prune.
We demonstrate our approach with extensive experiments on vision transformers on the ImageNet classification task.
Our results demonstrate that LTMP achieves **state-of-the-art accuracy across reduction rates while requiring only a single fine-tuning epoch**, which is an order of magnitude faster than previous methods.

## TL;DR

| ![token pruning](https://maxim.bonnaerens.com/publication/ltmp/85051_prune.png) | + | ![token merging](https://maxim.bonnaerens.com/publication/ltmp/85051_merge.png) | = | ![token merging and pruning](https://maxim.bonnaerens.com/publication/ltmp/85051_layer_11.png) |
| - | - | - | - | - |

## Overview

An overview of our framework is shown below. Given any vision transformer, our approach adds  merging (LTM) and pruning (LTP) components with learned threshold masking modules in each transformer block between the Multi-head Self-Attention (MSA) and MLP components. Based on the attention in the MSA, importance scores for each token and similarity scores between tokens are computed.
Learned threshold masking modules then learn the thresholds that decide which tokens to prune and which ones to merge.

![framework overview](https://maxim.bonnaerens.com/publication/ltmp/ltmp_schematic_portrait_poster.png)

## Installation

```bash
pip install -e .
```

## Usage
This repository is based on the vision transformers of [🤗`timm`](https://github.com/huggingface/pytorch-image-models) (v0.9.5).

### Training
LTMP Vision Transformers for training can be used as follows:
```python
import timm
import ltmp

model = timm.create_model("ltmp_vit_base_patch16_224", pretrained=True, tau=0.1, **kwargs)
```


To reproduce the results from the paper run:


```bash
python tools/train.py /path/to/imagenet/ --model ltmp_deit_small_patch16_224 --pretrained -b 128 --lr 0.000005 0.005 --reduction-target 0.75
```

### Inference
`"ltmp_{vit_model}"` models obtained through the training detailed above can be used for inference by using the following variant which actually prunes and merges tokens.

```python
import timm
import ltmp

model = timm.create_model("inference_ltmp_vit_base_patch16_224", pretrained=True, tau=0.1, **kwargs)
```

To check the accuracy of trained models:
```bash
python tools/validate_timm.py /path/to/imagenet/ --model ltmp_deit_small_patch16_224 --checkpoint /path/to/checkpoint.pth.tar -b 1
```

### Adapt to other vision transformers
See [`./ltmp/timm/lt_mergeprune.py`](./ltmp/timm/lt_mergeprune.py) and [`./ltmp/timm/lt_mergeprune_inference.py`](./ltmp/timm/lt_mergeprune_inference.py) for the changes required to adopt LTMP in a vision transformer.
[`./tools/train.py`](./tools/train.py) contains the code to train LTMP models.

## Citation
If you find this work useful, consider citing it:
```bibtex
@article{
    bonnaerens2023learned,
    title={Learned Thresholds Token Merging and Pruning for Vision Transformers},
    author={Maxim Bonnaerens and Joni Dambre},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=WYKTCKpImz},
    note={}
}
```
