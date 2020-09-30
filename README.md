# Neural Twins Talk

![teaser results](Demo.jpg)

## Introduction
We provide training and evaluation for both the NTT model (Zohourianshahzadi & Kalita, IEEE HCCAI 2020) and the NBT model (Jiasen Lu et al. IEEE CVPR 2018). 

## Project Environment Setup
First download the repository and download and unzip the [data.zip](https://drive.google.com/file/d/1265uL4btDgGRGqExR4s3kANBUWGDs9Fv/view) 
and [tools.zip](https://drive.google.com/file/d/1reAJwPnY6QTi5b5ixmA29uPrwbF_RDT5/view) inside the project directory.

This repository contains a Dockerfile for setting up the docker container for COCO experiments (Karpathy's / robust / Novel Splits) on GPU. To build the Docker container, execute
the following command from the project root:

```shell
docker build -t ntt .
```

Before running the container, you need to get COCO dataset downloaded and kept somewhere in your filesystem.
In order to do this, go to data folder, copy coco_2014.sh into a directory that is not a subset of project directory. For instance, inside bash do:

```shell
cd ../..
```

inside bash when in data folder, and then:

```shell
mv neuraltwinstalk/data/coco_2014.sh .
```

and then run the coco_2014.sh with bash:

```shell
sh coco_2014.sh
```

Declare two environment variables:

1. `$COCO_I`: path to a directory with sub-directories of images as `train2014`, `val2014`, `test2015`, etc...
2. `$COCO_A`: path to a directory with annotation files like `instances_train2014.json`, `captions_train2014.json` etc...

These directories will be attached as "volumes" to our docker container for Neural Twins Talk to use within. Get [nvidia-docker](https://www.github.com/NVIDIA/nvidia-docker) and execute this command to run the fresh built docker image.

```shell
nvidia-docker run --name ntt_container -it \
     -v $COCO_I:/workspace/neuraltwinstalk/data/coco/images \
     -v $COCO_A:/workspace/neuraltwinstalk/data/coco/annotations \
     --shm-size 16G -p 8888:8888 ntt /bin/bash
```

Ideally, shared memory size (`--shm-size`) of 16GB would be enough. Tune it according to your requirements / machine specifications.

**Saved Checkpoints:** All checkpoints will be saved in `/workspace/neuraltwinstalk/save`. From outside the container, execute this to get your checkpoints from this container into the main filesystem:
The container would expose port 8888, which can be used to host tensorboard visualizations.

```shell
docker container cp ntt_container:workspace/neuraltwinstalk/save /path/to/local/filesystem/save
```

Skip directly to **Training and Evaluation** section to execute specified commands within the container.


## Requirements

- Python 3.7
- [pytorch](http://pytorch.org/) : pytorch:0.4-cuda9-cudnn7-devel
- Other requirements are handled by DockerFile.

## Training and Evaluation (NTT)
### Data Preparation
All data is prepared and ready inside [data.zip](https://drive.google.com/file/d/1265uL4btDgGRGqExR4s3kANBUWGDs9Fv/view).

Next, go to prepro folder with bash and execute the following command: (This downloads the Stanford-corenlp, the version we need)

```shell
sh download_scnlp.sh
```

### Pretrained model

Pre-trained models will be available here soon. Stay tuned.

### Standard Image Captioning
##### Training (COCO)

First, modify the cofig file `cfgs/normal_coco_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True
```

##### Evaluation (COCO)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/normal_coco_1024_adam --mGPUs True --glove_6B_300 True
```

##### Training (Flickr30k)
Modify the cofig file `cfgs/normal_flickr_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/normal_flickr_res101.yml --batch_size 80 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True
```

##### Evaluation (Flickr30k)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/normal_flickr_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/normal_flickr30k_1024_adam --mGPUs True --glove_6B_300 True
```

### Robust Image Captioning

##### Training
Modify the cofig file `cfgs/normal_flickr_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/robust_coco.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True
```
##### Evaluation (robust-coco)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/robust_coco.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/robust_coco_1024 --mGPUs True --glove_6B_300 True
```

### Novel Object Captioning

##### Training

```
python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True
```
##### Evaluation (noc-coco)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/noc_coco_1024_adam --mGPUs True --glove_6B_300 True
```

## Training and Evaluation (NBT)
### Data Preparation
All data is prepared and ready inside [data.zip](https://drive.google.com/file/d/1265uL4btDgGRGqExR4s3kANBUWGDs9Fv/view).

Next, go to prepro folder with bash and execute the following command: (This downloads the Stanford-corenlp, the version we need)

```shell
sh download_scnlp.sh
```

### Pretrained model

Pre-trained models will be available here soon. Stay tuned.

### Standard Image Captioning
##### Training (COCO)

First, modify the cofig file `cfgs/normal_coco_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True --att_model topdown
```

##### Evaluation (COCO)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/normal_coco_1024_adam --mGPUs True --glove_6B_300 True --att_model topdown
```

##### Training (Flickr30k)
Modify the cofig file `cfgs/normal_flickr_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/normal_flickr_res101.yml --batch_size 80 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True --att_model topdown
```

##### Evaluation (Flickr30k)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/normal_flickr_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/normal_flickr30k_1024_adam --mGPUs True --glove_6B_300 True --att_model topdown
```

### Robust Image Captioning

##### Training
Modify the cofig file `cfgs/normal_flickr_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/robust_coco.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True --att_model topdown
```
##### Evaluation (robust-coco)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/robust_coco.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/robust_coco_1024 --mGPUs True --glove_6B_300 True --att_model topdown
```

### Novel Object Captioning

##### Training

```
python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --glove_6B_300 True --att_model topdown
```
##### Evaluation (noc-coco)
Train the model or Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/noc_coco_1024_adam --mGPUs True --glove_6B_300 True --att_model topdown
```

### Multi-GPU Training
For multiple GPU training simply add `--mGPUs Ture` in the command when training the model.

## Demo With detection bbox

For Karpathy's split on COCO you can run the following:

```
python demo.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/normal_coco_1024_adam4 --mGPUs True --glove_6B_300 True
```

For other splits, replace the main.py with demo.py in the evaluation commands.

## BERT Embeddings
We also provide the ability to train with BERT Embeddings, the original embedding used in experiences in the paper is Glove_6B_300.
Feel free to train the models with BERT embeddings as well. In order to train the models with BERT embeddings, simply replace the --glove_6B_300 in the commands to --bert_base_768. Similar to the following:

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --mGPUs True --bert_base_768 True
```

## Acknowledgement
We thank Jiasen Lu et al. for [NBT](https://github.com/jiasenlu/NeuralBabyTalk) repo and Ruotian Luo for his [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) repo. 
