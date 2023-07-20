# Learned Thresholds Token Merging and Pruning for Vision Transformers 
## [[`ES-FoMo @ICML2023`](https://openreview.net/forum?id=19pi10cY8x)] [[`arXiv`]()] [[`Project site`](maxim.bonnaerens.com/publication/ltmp)]

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

| ![token pruning](maxim.bonnaerens.com/publication/ltmp/85051_prune.png) | + | ![token merging](maxim.bonnaerens.com/publication/ltmp/85051_merge.png) | = | ![token merging and pruning](maxim.bonnaerens.com/publication/ltmp/85051_layer_11.png) |
| - | - | - | - | - |

## Overview

An overview of our framework is shown below. Given any vision transformer, our approach adds  merging (LTM) and pruning (LTP) components with learned threshold masking modules in each transformer block between the Multi-head Self-Attention (MSA) and MLP components. Based on the attention in the MSA, importance scores for each token and similarity scores between tokens are computed.
Learned threshold masking modules then learn the thresholds that decide which tokens to prune and which ones to merge.
![framework overview](maxim.bonnaerens.com/publication/ltmp/ltmp_schematic_portrait_poster.png)

## Installation

```bash
pip install -e .
```

## Training
To reproduce the results from the paper run:

```bash
python tools/train.py /path/to/imagenet/ --model ltmp_deit_small_patch16_224 --pretrained -b 128 --lr 0.000005 0.005 --reduction-target 0.75
```

## Citation
If you find this work useful, consider citing it:
```bibtex

```
