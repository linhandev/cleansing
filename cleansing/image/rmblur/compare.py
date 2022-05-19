import argparse
import os.path as osp

from PIL import Image
import numpy as np
import paddle
from paddle.vision.models import resnet50
import cv2
from SSIM_PIL import compare_ssim

from ..util import listdir

def parse_args():
    parser = argparse.ArgumentParser(description="Predict with SSIM model")
    parser.add_argument("--dataset_path", "-d", type=str, help="Path to dataset")
    parser.add_argument("--weight_path", "-w", type=str, default="./model/ssim.pdparams")
    parser.add_argument("--ksize", "-k", type=int, default=5)
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    model = paddle.Model(resnet50(pretrained=False, num_classes=1))
    model.load(args.weight_path)
    model.prepare()

    for fname in listdir(args.dataset_path):
        fpath = osp.join(args.dataset_path, fname)

        img = Image.open(fpath).convert("RGB")  # hwc bgr
        img_noise = cv2.blur(np.asarray(img), (args.ksize, args.ksize))
        img_noise_pil = Image.fromarray(img_noise.astype("uint8"))

        ssim = compare_ssim(img, img_noise_pil, GPU=False)

        img_noise = np.transpose(img_noise, [2, 0, 1]).astype("float32")
        img_noise = (img_noise - 255 / 2) / 255
        res = model.predict([np.array([img_noise])])[0][0].flatten()[0]
        print(res, ssim)


    print("Predicted ssim is: ", res)
