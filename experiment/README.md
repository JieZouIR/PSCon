#  PSCon: Product Search Through Conversations

## Requirements

This code is written in PyTorch. Any version later than 1.6 is expected to work with the provided code. Please refer to the [official website](https://pytorch.org/) for an installation guide.

We recommend to use conda for installing the requirements. If you haven't installed conda yet, you can find instructions [here](https://www.anaconda.com/products/individual). The steps for installing the requirements are:

+ Create a new environment

   ```
   conda create env -n PSCon
   ```

   In the environment, a python version >3.6 should be used.

+ Activate the environment

   ```
   conda activate PSCon
   ```

+ Install the requirements within the environment via pip:

   ```
   pip install -r requirements.txt
   ```

## Datasets

We use [KdConv](https://github.com/thu-coai/KdConv/tree/master/data) and [DuConv](https://ai.baidu.com/broad/introduction?dataset=duconv) datasets for pretraining. You can get them from the provided links and put them in the corresponding folders in `./data/`. For example, KdConv datasets should be put in `./data/KdConv`.  We use the PSCon dataset to fine-tune the model, and this dataset is available in `./data/PSCon`. Details about the PSCon dataset can be found [here](../README.md)'s Dataset Description.

## Training

+ Run the following scripts to automatically process the pretraining datasets into the required format:

```
python ./Run.py --mode='data'
```

+ Run the following scripts sequentially:

```
python -m torch.distributed.launch --nproc_per_node=4 ./Run.py --mode='pretrain'
python -m torch.distributed.launch --nproc_per_node=4 ./Run.py --mode='finetune'
```

Note that you should select the appropriate pretrain models from the folder `./output/pretrained`, and put them into `./output/pretrained_ready` which is newly created by yourself before finetuning. The hyperparameters are set to the default values used in our experiments. To see an overview of all hyperparameters, please refer to `./Run.py`.

## Evaluating

+ Run the following scripts:

```
python -m torch.distributed.launch --nproc_per_node=4 ./Run.py --mode='infer-valid'
python -m torch.distributed.launch --nproc_per_node=4 ./Run.py --mode='eval-valid'
```

```
python -m torch.distributed.launch --nproc_per_node=4 ./Run.py --mode='infer-test'
python -m torch.distributed.launch --nproc_per_node=4 ./Run.py --mode='eval-test'
```
