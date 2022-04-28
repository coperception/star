# Multi-Agent-Autoencoder

We implement multi-agent encoder: the input is a single-view voxelized point cloud, and the output is a multi-view voxelized point cloud. There is information fusion at the intermediate feature layer/codeword. Our main target is to recover the complete scene via communicating a small-sized latent feature.

utils/CoDetModule.py (def step(self,data,batch_size))

utils/models/backbone/Backbone.py 

utils/models/base/DetModelBase.py

utils/models/base/FusionBase.py

utils/models/base/IntermediateModelBase.py



## NOTE:

Currently multi-gpu runtime is not supported.

If you have more than 1 gpu in your environment, Please specify a gpu for the following bash commands.

```bash
EXPORT CUDA_VISIBLE_DEVICES={Your_desired_gpu_number}
```



## Preparation

- Please download the training/val set [**V2X-Sim-1.0-trainval**](https://drive.google.com/file/d/11lyIaOeNMCpJkZDOydxqGBNoHiTxTgZk/view?usp=sharing).

- Set dataset folder in config.yaml

  - ```
    data: {PATH_TO_DATASET}/V2X-Sim-1.0-trainval
    ```

## Training

Train multi-agent-autoencoder:

Modify config.yaml for setting training parameters. 

```bash
# sum-fusion
python train_mae.py
```

In training process, if log option is True, all the running codes and config files will be copied to the **logpath** specified in config. yaml 

training log will be saved in {logpath}/log.txt



## Evaluation

Evaluate multi-agent-encoder:

```bash
python eval_mae.py --resume $PATH_TO_MAE_CKPT
```

This script will automatically search for training args and model state dict, as well as eval args in log path.

 

For example, 

```bash
python eval_mae.py --resume logs/test_mae/sum/epoch_20.pth
```

Then,

eval_mae.py will load training and evaluation args in 

```bash
logs/test_mae/sum/config.yaml
```



To modify evaluation args such as metrics, please refer to **config.yaml** in **logpath**.

Eval result will be saved in 

```bash
{logpath}/eval/Epoch_{epoch_num}
```





## Train a multi-agent detector

To train multi-agent detection using the result outputted by Multi-Agent AutoEncoder:

```bash
python train_codet.py --data {PATH_TO_DATASET}/V2X-Sim-1.0-trainval/train --log --logpath $LOGPATH --resume_mae $PATH_TO_MAE_CKPT --layer layer_num_sequence
```



```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  The path to the preprocessed sparse BEV training data
  --batch BATCH         Batch size
  --nepoch NEPOCH       Number of epochs
  --nworker NWORKER     Number of workers
  --lr LR               Initial learning rate
  --log                 Whether to log
  --logpath LOGPATH     The path to the output log file
  --resume RESUME       The path to the saved model that is loaded to resume
                        training
  --resume_mae RESUME_MAE
                        The path to the saved MAE model that is loaded to
                        resume training
  --layer LAYER         Communicate which layer in the single layer com mode
  --warp_flag           Whether to use pose info for Ｗhen2com
  --kd_flag KD_FLAG     Whether to enable distillation (only DiscNet is 1 )
  --kd_weight KD_WEIGHT
                        KD loss weight
  --gnn_iter_times GNN_ITER_TIMES
                        Number of message passing for V2VNet
  --visualization VISUALIZATION
                        Visualize validation result
  --com COM             disco/when2com/v2v/sum/mean/max/cat/agent
  --bound BOUND         The input setting: lowerbound -> single-view or
                        upperbound -> multi-view
  --output_thresh OUTPUT_THRESH
                        Output threshold for mae
  --finetune            Whether to finetune auto-encoder
```



#### Some important args:

- resume_mae: Path to the multi-agent auto encoder checkpoint.

- log path: shoule be specified if you want to save the checkpoint. Normally it should be

  ```bash
  logs/{Your_desired_log_folder_name}
  ```

- **layer: Sequence of Intergers corresbonding with communating layer number as the same in config.yaml of multi-agent autoencoder**

  - For example, if the layer arg in config.yaml in target MAE log path is 

    ```yaml
    layer: 
    #layer num for communication
      - 1
      - 2
      # - 3
    ```

    Then it should be

    ```bash
    --layer 1 2 
    ```

    in bash command line.

- output_thresh: Affects the binary output threshold for Multi-Agent AutoEncoder as the input for detector.

- finetune: if specified, gradient of training a detector will back propagate to params in Multi-Agent AutoEncoder as a process of finetuning the MAE. Finetuned checkpoint will be saved in

  ```bash
  $LOGPATH/epoch_{epoch_num}_finetune_ae.pth
  ```

  The checkpoint of detector will be saved in

  ```bash
  $LOGPATH/epoch_{epoch_num}_finetune_det.pth
  ```

  if finetune option is not specified, only the checkpoint of detector will be saved in 

  ```
  $LOGPATH/epoch_{epoch_num}_det.pth
  ```

  

#### Example:

```bash
python train_codet.py --data /mnt/NAS/home/qifang/datasets/V2X-Sim-1.0-trainval/train --log --logpath logs/test_det --resume_mae logs/test_mae/sum/epoch_20.pth --layer 2 --output_thresh 0.1 --finetune
```





## Eval a multi-agent detector

To eval multi-agent detection using the result outputted by Multi-Agent AutoEncoder:

```bash
python test_codet.py --data {PATH_TO_DATASET}/V2X-Sim-1.0-trainval/test --log --resume_covae $PATH_TO_MAE_EVAL_CKPT --resume $PATH_TO_DET_CKPT --layer [layer_num_list]
```



```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  The path to the preprocessed sparse BEV training data
  --batch BATCH         Batch size
  --nepoch NEPOCH       Number of epochs
  --nworker NWORKER     Number of workers
  --lr LR               Initial learning rate
  --log                 Whether to log
  --resume RESUME       The path to the saved model that is loaded to resume
                        training
  --resume_mae RESUME_MAE
                        The path to the saved MAE model that is loaded to
                        resume training
  --layer LAYER         Communicate which layer in the single layer com mode
  --warp_flag           Whether to use pose info for Ｗhen2com
  --kd_flag KD_FLAG     Whether to enable distillation (only DiscNet is 1 )
  --kd_weight KD_WEIGHT
                        KD loss weight
  --gnn_iter_times GNN_ITER_TIMES
                        Number of message passing for V2VNet
  --visualization VISUALIZATION
                        Visualize validation result
  --com COM             disco/when2com/v2v/sum/mean/max/cat/agent
  --bound BOUND         The input setting: lowerbound -> single-view or
                        upperbound -> multi-view
  --inference INFERENCE
  --tracking
  --box_com
  --output_thresh OUTPUT_THRESH
                        Output threshold for mae
  --log_tag LOG_TAG     log tag for eval
```



#### Some important args:

- resume_mae: Path to the multi-agent auto encoder checkpoint.
- resume_det: Path to the detector checkpoint
- output_thresh: Affects the binary output threshold for Multi-Agent AutoEncoder as the input for detector.
- log_tag: Adds lag for log folder.



Evaluation results will be saved in

```bash
$PATH_TO_DET_CKPT/lowerbound/eval/epoch_{epoch_num}{_[log_tag](if specified)}
```



#### Example:

```bash
python test_codet.py --data /mnt/NAS/home/qifang/datasets/V2X-Sim-1.0-trainval/test --log  --resume_mae logs/test_det/epoch_100_finetune_ae.pth --resume logs/test_det/epoch_100_finetune_det.pth --layer 2 --output_thresh 0.1
```

