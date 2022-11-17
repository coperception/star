# Multi Robot Scene Completion: Towards Task-agnostic Collaborative Perception

This readme currently work in progress.

* [ ] a link to our paper

![main](https://github.com/coperception/star/raw/gh-pages/static/images/main.jpg)

## News:

**[2022-10]** The project [website](https://coperception.github.io/star/) is online.

**[2022-09]** Our work is accepted at the **6th Conference on Robot Learning (CoRL 2022)**.

## Abstract:

Collaborative perception learns how to share information among multiple robots to perceive the environment better than individually done. Past research on this has been task-specific, such as detection or segmentation. Yet this leads to different information sharing for different tasks, hindering the large-scale deployment of collaborative perception. We propose the first task-agnostic collaborative perception paradigm that learns a single collaboration module in a self-supervised manner for different downstream tasks. This is done by a novel task termed multi-robot scene completion, where each robot learns to effectively share information for reconstructing a complete scene viewed by all robots. Moreover, we propose a spatiotemporal autoencoder (STAR) that amortizes over time the communication cost by spatial sub-sampling and temporal mixing. Extensive experiments validate our method's effectiveness on scene completion and collaborative perception in autonomous driving scenarios.

## Usage:

Our code is based on the [coperception](https://coperception.readthedocs.io/en/latest/) library. Currently we have included the library in this repository for direct reproducibility convenience. You can run the following command to have the library installed.

```bash
pip install -e .
```

The work is tested with:

* python 3.7
* pytorch 1.8.0
* torchvision 0.9.1
* timm 0.3.2

To train, run:

```bash
cd tools/scene_completion/
make train_completion
```

To test the trained model on scene completion:

```bash
cd tools/scene_completion/
make test_completion
```

More commands and experiment settings are included in the [Makefile](https://github.com/coperception/star/raw/main/tools/scene_completion/Makefile).

You can find the training and test scripts at: [tools/scene_completion](https://github.com/coperception/star/raw/main/tools/scene_completion/).


## Citation:

```
@inproceedings{liself,
  title={Multi-Robot Scene Completion: Towards Task-Agnostic Collaborative Perception},
  author={Li, Yiming and Zhang, Juexiao and Ma, Dekun and Wang, Yue and Feng, Chen},
  booktitle={6th Annual Conference on Robot Learning}
}
```
