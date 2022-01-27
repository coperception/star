# Multi-Agent-Autoencoder

We implement lowerbound, upperbound, when2com, who2com, V2VNet as our benchmark detectors. Please see more details in our paper.

## Preparation

- Please download the training/val set [**V2X-Sim-1.0-trainval**](https://drive.google.com/file/d/11lyIaOeNMCpJkZDOydxqGBNoHiTxTgZk/view?usp=sharing).

## Training

Train multi-agent-autoencoder:
*PATH_TO_DATA is where you put your data*

```bash
# sum-fusion
python train_mae.py --data PATH_TO_DATA --bound lowerbound --com sum --log

```

## Evaluation

Evaluate multi-agent-encoder:

```bash
# lowerbound
Please finish the test script.

```
