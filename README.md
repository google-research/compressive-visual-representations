# Compressive Visual Representations

This repository contains the source code for our paper,
[Compressive Visual Representations](https://arxiv.org/abs/2109.12909).
We developed information-compressed versions of the SimCLR and BYOL
self-supervised learning algorithms, which we call C-SimCLR and C-BYOL, using
the Conditional Entropy Bottleneck, and achieved significant improvements in
accuracy and robustness, yielding linear evaluation performance competitive with
fully supervised models.

![cvr_perf](https://user-images.githubusercontent.com/4847452/144777161-d8d5ec7a-dc4e-4f45-a9c2-f57c2661d8eb.png)


We include implementations of the C-SimCLR and C-BYOL algorithms developed in
our paper, as well as SimCLR and BYOL baselines.

## Getting Started

Install the necessary dependencies with `pip install -r requirements.txt`.
We recommend creating a new virtual environment.

To train a model with C-SimCLR on ImageNet run
`bash scripts/csimclr.sh`. And to train a model with C-BYOL, run
`bash scripts/cbyol.sh`.

Refer to the scripts for further configuration options, and also to train the
corresponding SimCLR and BYOL baselines.

These command lines use the hyperparameters used to train the models in our
paper. In particular, we used a batch size of 4096 using 32 Cloud TPUs.
Using different accelerators will require reducing the batch size.
To get started with Google Cloud TPUs, we recommend following this
[tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist).

## Checkpoints

The following table contains pretrained checkpoints for C-SimCLR, C-BYOL and
also their respective baselines, SimCLR and BYOL. All models are trained on
ImageNet. The Top-1 accuracy is obtained by training a linear classifier on top
of a ``frozen'' backbone whilst performing self-supervised training of the
network.

| Algorithm | Backbone     | Training epochs | ImageNet Top-1 | Checkpoint |
|-----------|:------------:|:---------------:|:--------------:|:-----:|
| SimCLR    | ResNet 50    | 1000            | 71.1           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/simclr/resnet50/checkpoint.tar.gz)      |
| SimCLR    | ResNet 50 2x | 1000            | 74.6           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/simclr/resnet50-2x/checkpoint.tar.gz)      |
| C-SimCLR  | ResNet 50    | 1000            | 71.8           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/simclr/resnet50/checkpoint.tar.gz)      |
| C-SimCLR  | ResNet 50 2x | 1000            | 74.7           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/simclr/resnet50-2x/checkpoint.tar.gz)      |
| BYOL      | ResNet 50    | 1000            | 74.4           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/byol/resnet50/checkpoint.tar.gz)      |
| BYOL      | ResNet 50 2x | 1000            | 77.3           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/byol/resnet50-2x/checkpoint.tar.gz)      |
| C-BYOL    | ResNet 50    | 1000            | 75.9           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/cbyol/resnet50/1000_epochs/checkpoint.tar.gz)      |
| C-BYOL    | ResNet 50 2x | 1000            | 79.1           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/cbyol/resnet50-2x/checkpoint.tar.gz)      |
| C-BYOL    | ResNet 101   | 1000            | 78.0           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/cbyol/resnet101/checkpoint.tar.gz)      |
| C-BYOL    | ResNet 152   | 1000            | 78.8           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/cbyol/resnet152/checkpoint.tar.gz)      |
| C-BYOL    | ResNet 50    | 1500            | 76.0           | [link](https://storage.googleapis.com/rl-infra-public/compressive-visual-representations/checkpoints/cbyol/resnet50/1500_epochs/checkpoint.tar.gz)      |

## Reference

If you use C-SimCLR or C-BYOL, please use the following BibTeX entry.
```
@InProceedings{lee2021compressive,
  title={Compressive Visual Representations},
  author={Lee, Kuang-Huei and Arnab, Anurag and Guadarrama, Sergio and Canny, John and Fischer, Ian},
  booktitle={NeurIPS},
  year={2021}
}
```

## Credits

This repository is based on [SimCLR](https://github.com/google-research/simclr).
We also match our BYOL implementation in Tensorflow 2 to the original
implementation of
[BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol) in JAX.


*Disclaimer: This is not an official Google product.*
