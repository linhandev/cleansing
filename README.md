# cleansing

Cleansing is a dataset cleansing tool for machine learning model training.

We currently support clenaning image datasets. The cleansing measures implemented are:

- Deduplication
- Deredundant
- Identify blurry image

To use this tool, first clone the repo

```shell
git clone https://github.com/linhandev/cleansing
```
Download pretrained weights for deep learning models [here](https://drive.google.com/drive/folders/1GzYqK4idR7DuifIYhwkjFcI9BSOcVMkr?usp=sharing) and put them in cleansing/model folder.

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

## Deduplication

## Deredundant

## Identify blurry image

The pipeline is based on deep learning. The dataset is prepared with images downloaded from flicker. To download a dataset plz run:

```shell
python noise/download_data.py
```

note that flicker api should be run with python<=3.7!

No preprocessing or labeling is needed. To train the model run:

```shell
python cleansing/image/blur/train.py
```

Models will be saved in `saved_model` folder.

To run model on individal image run:

```shell
python cleansing/image/blur/infer.py path_to_the_image
```

To get rid of blurry images in a dataset run:

```shell
python cleansing/image/blur/deblur.py
```
