import argparse
import os.path as osp

import paddle

from .predict import predict
from paddle.vision.models import resnet50
from ..util import listdir

def parse_args():
    parser = argparse.ArgumentParser(description="Predict with SSIM model")
    parser.add_argument("--dataset_path", "-d", type=str, help="Path to image")
    parser.add_argument("--weight_path", "-w", type=str, default="./model/ckpt/final.pdparams")
    parser.add_argument("--thresh", "-t", default=0.5, type=float)
    args = parser.parse_args()
    return args

def rmblur(dataset_path, weight_path, thresh):
    model = paddle.Model(resnet50(pretrained=False, num_classes=1))
    model.load(weight_path, reset_optimizer=True)
    model.prepare()

    file_paths = [osp.join(dataset_path, n) for n in listdir(dataset_path)]

    blur_scores = []
    for path in file_paths:
        blur_scores.append(predict(model, path))
    
    to_remove = []
    for blur_score, file_path in zip(blur_scores, file_paths):
        if blur_score < thresh:
            to_remove.append(file_path)

    print("Suggest to remove:")
    for f in to_remove:
        print(f, end=" ")
    print()


if __name__ == "__main__":
    rmblur(**parse_args().__dict__)

