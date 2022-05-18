# cleansing

Cleansing is a tool for cleaning dataset used for machine learning model training.

We currently support clenaning image datasets. The cleansing measures implemented are:

- Deduplication: remove similar images
- Deredundant: remove outlier images
- Remove blurry image

To use this tool, first clone the repo

```shell
git clone https://github.com/linhandev/cleansing
```

Download pretrained weights for deep learning models [here](https://drive.google.com/drive/folders/1GzYqK4idR7DuifIYhwkjFcI9BSOcVMkr?usp=sharing) and put the .pdparams file in `model` folder.

(Optional) You can create a new environment to avoid dependency issues.

```shell
conda create -n clenasing python=3.9
conda create -n flickr python=3.7 # flickr should be run with python<=3.7
```

Then install dependencies:

```shell
pip install -r requirements.txt
```

Note that the deep learning part of this project is based on PaddlePaddle, it's not included in requirements.txt. See install instructions [here](https://www.paddlepaddle.org.cn/install/quick)

## Deduplicate

See parameter details with

```shell
python -m cleansing.image.deduplicate --help
```

Run deduplication with

```shell
python -m cleansing.image.deduplicate --dataset_path data/demo --cluster_number 1 --percentage 0.88
python -m cleansing.image.deduplicate --dataset_path data/demo --cluster_number 1 --thresh 0.0545
```

## Deredundant

See parameter details with

```shell
python -m cleansing.image.deredundant --help
```

Run deredundant with

```shell
python -m cleansing.image.deredundant --dataset_path data/demo --percentage 0.1
```

## Remove blurry image

This pipeline is based on deep learning. The dataset is prepared with images downloaded from flickr. To download a dataset run

```shell
python tool/flickr.py
```

note that flickr api should be run with python<=3.7. Flickr limit the number of api request an account can send to 3600 per hour. May need to run this script multiple times.

No preprocessing or labeling is needed. To train the model run:

```shell
python -m cleansing.image.rmblur.train --dataset_path data
```

Models will be saved under `model/ckpt` folder, use `model/ckpt/final.pdparams` during inference.

To run prediction on individal image run:

```shell
python cleansing.image.rmblur.predict -i /path/to/the/image
```

To remove blurry images in a dataset run:

```shell
python -m cleansing.image.rmblur --dataset_path /path/to/dataset --thresh 0.9
```
