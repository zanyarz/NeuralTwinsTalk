
FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

COPY . /workspace/neuraltwinstalk

# ----------------------------------------------------------------------------
# -- install apt and pip dependencies
# ----------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y \
    ant \
    vim \
    ca-certificates-java \
    nano \
    openjdk-8-jdk \
    python2.7 \
    unzip \
    wget && \
    apt-get clean

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN update-ca-certificates -f && export JAVA_HOME

RUN pip install --upgrade pip

RUN pip install Cython && pip install h5py \
    setuptools==41.0.0 \
    matplotlib \
    nltk \
    mxnet-cu90\
    bert-embedding \
    gluonnlp \
    numpy==1.14.6 \
    pycocotools \
    scikit-image \
    stanfordcorenlp \
    tensorflow \
    torchtext \
    tqdm && python -c "import nltk; nltk.download('punkt')"

# ----------------------------------------------------------------------------
# -- download Karpathy's preprocessed captions datasets and corenlp jar
# ----------------------------------------------------------------------------

RUN cd /workspace/neuraltwinstalk/data && \
    wget --quiet http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip && \
    unzip caption_datasets.zip && \
    mv dataset_coco.json coco/ && \
    mv dataset_flickr30k.json flickr30k/ && \
    rm caption_datasets.zip dataset_flickr8k.json

RUN cd /workspace/neuraltwinstalk/prepro && \
    wget --quiet https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip && \
    unzip stanford-corenlp-full-2017-06-09.zip && \
    rm stanford-corenlp-full-2017-06-09.zip

RUN cd /workspace/neuraltwinstalk/tools/coco-caption && \
    sh get_stanford_models.sh

# ----------------------------------------------------------------------------
# -- download preprocessed COCO detection output HDF file and pretrained model
# ----------------------------------------------------------------------------

RUN cd /workspace/neuraltwinstalk/data/coco && \
    wget --quiet https://www.dropbox.com/s/2gzo4ops5gbjx5h/coco_detection.h5.tar.gz && \
    tar -xzvf coco_detection.h5.tar.gz && \
    rm coco_detection.h5.tar.gz


WORKDIR /workspace/neuraltwinstalk
RUN python prepro/prepro_dic_coco.py \
    --input_json data/coco/dataset_coco.json \
    --split normal \
    --output_dic_json data/coco/dic_coco.json \
    --output_cap_json data/coco/cap_coco.json && \
    python prepro/prepro_dic_coco.py \
    --input_json data/coco/dataset_coco.json \
    --split robust \
    --output_dic_json data/robust_coco/dic_coco.json \
    --output_cap_json data/robust_coco/cap_coco.json && \
    python prepro/prepro_dic_coco.py \
    --input_json data/coco/dataset_coco.json \
    --split noc \
    --output_dic_json data/noc_coco/dic_coco.json \
    --output_cap_json data/noc_coco/cap_coco.json

EXPOSE 8888
