# Multi-Level Knowledge Distillation

## Introduction

This is the GitHub repo for MLKD - a knowledge distillation technique which can produce state-of-the-art results for several image classification problems.
The table below shows the top-1 accuracies for various ResNet-50 students trained using MLKD:

| Dataset         | Accuracy |
|-----------------|----------|
| CIFAR-10        | 97.79    |
| CIFAR-100       | 86.88    |
| Oxford-IIIT Pet | 93.32    |

[//]: # (TODO: Add link to download models)
[//]: # (TODO: Add link to paper)

## Dependencies
* Python 3.9
* PyTorch 2.0.1
* torchvision 0.15.2
* timm 0.9.2

## Building an Ensemble (optional)

Whilst any teacher can be trained using this method, optimal distillation is achieved using an ensemble teacher.

Example config.json:
```json
{
    "models": ["deit_small_distilled_patch16_224", "convnextv2_nano", "gmlp_s16_224", "hrnet_w18", "swinv2_cr_tiny_ns_224"],
    "names": ["oxford_deit", "e_convnext", "e_gmlp", "e_hrnet", "e_swin"],
    "weights": [0.9736, 1.0213, 0.9505, 0.9710, 0.9423]
}
```

## Training a ResNet-50 Student

Choices of dataset include `cifar-10`, `cifar-100`, and `oxford-iiit`.
Models will be saved in the models/ directory.
To run of multiple GPUs, pass the `--world-size` argument.
If training using a single teacher model, pass the `--checkpoint` and `--model` arguments to specify the checkpoint path and the timm model name of the teacher, respectively.

```commandline
python3 mlkd.py -b 128 -d oxford-iiit -n student_model --checkpoint models/resnet152.pth --model resnetv2_152x2_bit
```

If training using an ensemble model instead, pass the `--ensemble` argument.

```commandline
python3 mlkd.py -b 128 -d oxford-iiit -n student_model --ensemble config.json
```

## Loading Pre-Trained Models

As part of the project, ResNet-50 models were trained on CIFAR-100, Oxford-IIIT Pet. To load one of these pre-trained models:
```python
import timm

model = timm.create_model('resnetv2_50x1_bit', checkpoint_path=path_to_model, num_classes=num_classes)
```
