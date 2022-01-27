# Multi-Agent-Autoencoder

We implement multi-agent encoder: the input is a single-view voxelized point cloud, and the output is a multi-view voxelized point cloud. There is information fusion at the intermediate feature layer/codeword. Our main target is to recover the complete scene via communicating a small-sized latent feature.

utils/CoDetModule.py (def step(self,data,batch_size))

utils/models/backbone/Backbone.py 

utils/models/base/DetModelBase.py

utils/models/base/FusionBase.py

utils/models/base/IntermediateModelBase.py


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
