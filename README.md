# ConvNext-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545v2.pdf).

## Table of contents

- [ConvNext-PyTorch](#convnext-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [A ConvNet for the 2020s](#a-convnet-for-the-2020s)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `convnext_tiny`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 91: `model_weights_path` change to `./results/pretrained_models/ConvNext_tiny-ImageNet_1K-b03a77c2.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `convnext_tiny`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 51: `pretrained_model_weights_path` change
  to `./results/pretrained_models/ConvNext_tiny-ImageNet_1K-b03a77c2.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `convnext_tiny`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 54: `resume` change to `./samples/convnext_tiny-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/2201.03545v2.pdf](https://arxiv.org/pdf/2201.03545v2.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|     Model      |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:--------------:|:-----------:|:-----------------:|:-----------------:|
| convnext_tiny  | ImageNet_1K | 17.9%(**17.5%**)  |    -(**3.9%**)    |
| convnext_small | ImageNet_1K | 16.9%(**16.4%**)  |    -(**3.4%**)    |
| convnext_base  | ImageNet_1K | 15.9%(**15.9%**)  |    -(**3.1%**)    |
| convnext_large | ImageNet_1K | 14.5%(**15.6%**)  |    -(**3.0%**)    |

```bash
# Download `ConvNext_tiny-ImageNet_1K-b03a77c2.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `convnext_tiny` model successfully.
Load `convnext_tiny` model weights `/ConvNext-PyTorch/results/pretrained_models/ConvNext_tiny-ImageNet_1K-b03a77c2.pth.tar` successfully.
tench, Tinca tinca                                                          (38.61%)
barracouta, snoek                                                           (2.95%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.53%)
reel                                                                        (0.52%)
croquet ball                                                                (0.36%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### A ConvNet for the 2020s

*Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie*

##### Abstract

The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly
superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces
difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the
hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers
practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision
tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of
Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces
and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of
a vision Transformer, and discover several key components that contribute to the performance difference along the way.
The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard
ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8%
ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining
the simplicity and efficiency of standard ConvNets.

[[Paper]](https://arxiv.org/pdf/2201.03545v2.pdf)

```bibtex
@inproceedings{liu2022convnet,
  title={A convnet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11976--11986},
  year={2022}
}
```