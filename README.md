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

Download pretrained weights for deep learning models [here](https://drive.google.com/drive/folders/1GzYqK4idR7DuifIYhwkjFcI9BSOcVMkr?usp=sharing) and put them in `cleansing/model` folder.

Please update the project, some features were being finalized when submitting this version.

```shell
git pull
```

(Optional) You can create a new environment to avoid dependency issues.

```shell
conda create -n clenasing python=3.9
conda create -n flicker python=3.7 # flicker should be run with python<=3.7
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
python -m cleansing.image.deduplicate --dataset_path /path/to/dataset --percentage 0.9 --hashes ahash phash
```

## Deredundant

See parameter details with

```shell
python -m cleansing.image.deredundant -h
```

Run deredundant with

```shell
python -m cleansing.image.deredundant --dataset_path /path/to/dataset --isolation_thresh 0.9 --hashes ahash phash
```

## Identify blurry image

This pipeline is based on deep learning. The dataset is prepared with images downloaded from flicker. To download a dataset run

```shell
python noise/download_data.py
```

note that flicker api should be run with python<=3.7! Flicker have quota on number of images an account can download during an hour. Need to run this script multiple times.

No preprocessing or labeling is needed. To train the model run:

```shell
python cleansing/image/blur/train.py
```

Models will be saved in `clenasing/model` folder.

To run prediction on individal image run:

```shell
python cleansing/image/blur/infer.py /path/to/the/image
```

To remove blurry images in a dataset run:

```shell
python -m cleansing.image.rmblur --dataset_path /path/to/dataset --thresh 0.9
```
