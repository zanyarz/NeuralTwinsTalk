# Neural Twins Talk

![teaser results](Demo.jpg)


## Docker Setup
First download the repository and download and unzip the [data.zip](https://drive.google.com/file/d/1265uL4btDgGRGqExR4s3kANBUWGDs9Fv/view) 
and [tools.zip](https://drive.google.com/file/d/1reAJwPnY6QTi5b5ixmA29uPrwbF_RDT5/view) inside the project directory.

This repository contains a Dockerfile for setting up the docker container for COCO experiments (Karpathy's / robust / Novel Splits) on GPU. To build the Docker container, execute
the following command from the project root:

```shell
docker build -t nbt .
```

Before running the container, you need to get COCO dataset downloaded and kept somewhere in your filesystem.
Declare two environment variables:

1. `$COCO_I`: path to a directory with sub-directories of images as `train2014`, `val2014`, `test2015`, etc...
2. `$COCO_A`: path to a directory with annotation files like `instances_train2014.json`, `captions_train2014.json` etc...

These directories will be attached as "volumes" to our docker container for Neural Twins Talk to use within. Get [nvidia-docker](https://www.github.com/NVIDIA/nvidia-docker) and execute this command to run the fresh built docker image.

```shell
nvidia-docker run --name ntt_container -it \
     -v $COCO_I:/workspace/neuraltwinstalk/data/coco/images \
     -v $COCO_ANNOTATIONS:/workspace/neuraltwinstalk/data/coco/annotations \
     --shm-size 16G -p 8888:8888 nbt /bin/bash
```

Ideally, shared memory size (`--shm-size`) of 16GB would be enough. Tune it according to your requirements / machine specifications.

**Saved Checkpoints:** All checkpoints will be saved in `/workspace/neuralbabytalk/save`. From outside the container, execute this to get your checkpoints from this container into the main filesystem:
The container would expose port 8888, which can be used to host tensorboard visualizations.

```shell
docker container cp nbt_container:workspace/neuralbabytalk/save /path/to/local/filesystem/save
```

Skip directly to **Training and Evaluation** section to execute specified commands within the container.


## requirement

Inference:

- [pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- [torchtext](https://github.com/pytorch/text)

Data Preparation:

- [stanford-corenlp-wrapper](https://github.com/Lynten/stanford-corenlp)
- [stanford-corenlp](https://stanfordnlp.github.io/CoreNLP/)

Evaluation:

- [coco-caption](https://github.com/jiasenlu/coco-caption): Download the modified version of coco-caption and put it under `tools/`


## Demo

#### Without detection bbox


#### With detection bbox

#### Constraint beam search
This code also involve the implementation of constraint beam search proposed by Peter Anderson. I'm not sure my impmentation is 100% correct, but it works well in conjuction with neural baby talk code. You can refer to [this](http://users.cecs.anu.edu.au/~sgould/papers/emnlp17-constrained-beam-search.pdf) paper for more details. To enable CBS while decoding, please set the following flags:
```
--cbs True|False : Whether use the constraint beam search.
--cbs_tag_size 3 : How many detection bboxes do we want to include in the decoded caption.
--cbs_mode all|unqiue|novel : Do we allow the repetive bounding box? `novel` is an option only for novel object detection task.
```

## Training and Evaluation
### Data Preparation
Head to `data/README.md`, and prepare the data for training and evaluation.

### Pretrained model
| Task | Dataset | Backend | Batch size | Link  |
| ---- | :----:| :----:| :----:|:----:|
| Standard image captioning | COCO | Res-101 | 100 | [Pre-trained Model](https://www.dropbox.com/s/6buajkxm9oed1jp/coco_nbt_1024.tar.gz?dl=0) |
| Standard image captioning | Flickr30k | Res-101 | 50 | [Pre-trained Model](https://www.dropbox.com/s/cirzj1b2jul6yzx/flickr30k_nbt_1024.tar.gz?dl=0) |
| Robust image captioning | COCO | Res-101 | 100 | [Pre-trained Model](https://www.dropbox.com/s/sxuodvob0ftesm9/robust_coco_nbt_1024.tar.gz?dl=0) |
| Novel object captioning | COCO | Res-101 | 100 | [Pre-trained Model](https://www.dropbox.com/s/b7i6vx5pf98540l/noc_coco_nbt_1024.tar.gz?dl=0) |


### Standard Image Captioning
##### Training (COCO)

First, modify the cofig file `cfgs/normal_coco_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30
```
##### Evaluation (COCO)
Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/coco_nbt_1024
```

##### Training (Flickr30k)
Modify the cofig file `cfgs/normal_flickr_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/normal_flickr_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30
```

##### Evaluation (Flickr30k)
Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/normal_flickr_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/flickr30k_nbt_1024
```

### Robust Image Captioning

##### Training
Modify the cofig file `cfgs/normal_flickr_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/robust_coco.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30
```
##### Evaluation (robust-coco)
Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/robust_coco.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/robust_coco_nbt_1024
```

### Novel Object Captioning

##### Training
Modify the cofig file `cfgs/noc_coco_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30
```
##### Evaluation (noc-coco)
Download Pre-trained model. Extract the tar.zip file and put it under `save/`.

```
python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 20 --cuda True --num_workers 20 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/noc_coco_nbt_1024
```

### Multi-GPU Training
For multiple GPU training simply add `--mGPUs Ture` in the command when training the model.

## Acknowledgement
We thank Jiasen Lou et al. for [NBT] (https://github.com/jiasenlu/NeuralBabyTalk) repo and Ruotian Luo for his [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) repo. 
