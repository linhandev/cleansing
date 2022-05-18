import argparse

from PIL import Image
import numpy as np
import paddle
from paddle.vision.models import resnet50, vgg16, LeNet


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with SSIM model")
    parser.add_argument("--img_path", "-i", type=str, help="Path to image")
    parser.add_argument("--weight_path", "-w", type=str, default="./model/ssim")
    args = parser.parse_args()
    return args


def predict(model, img_path):

    img = Image.open(img_path)  # hwc bgr
    img = np.asarray(img)
    img = np.transpose(img, [2, 0, 1]).astype("float32")
    img = (img - 255/2) /255
    res = model.predict([img])
    return res

if __name__ == "__main__":
    args = parse_args()
    
    model = paddle.Model(resnet50(pretrained=True, num_classes=1))
    # model.load(args.weight_path)

    res = predict(model, args.img_path)

    print("Predicted ssim is: ", res)
