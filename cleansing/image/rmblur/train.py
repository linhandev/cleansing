import random
import cv2
import argparse
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
import paddle
from paddle.vision.models import resnet50, vgg16, LeNet
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from SSIM_PIL import compare_ssim

from ..util import listdir

# import matplotlib

# matplotlib.use("TkAgg")

# (N,3,H,W), bgr
class Dataset(paddle.io.Dataset):
    def __init__(self, data_dir, mode="train"):
        data_paths = listdir(data_dir)
        self.data_paths = [osp.join(data_dir, f) for f in data_paths]

        if mode == "train":
            self.length = len(self.data_paths)
        elif mode == "val":
            self.length = min(len(self.data_paths), 100)
        print(f"{mode} dataset contains {self.length} samples")

    def __getitem__(self, idx):
        img = Image.open(self.data_paths[idx]).convert("RGB")  # hwc rgb
        
        if img.size != (512, 512):
            img = img.resize((512, 512))

        # img_noise = skimage.util.random_noise(np.asarray(img), mode="gaussian")
        if random.random() > 0.05:
            ksize = int(random.random() * 10) + 1
            img_noise = cv2.blur(np.asarray(img), (ksize, ksize))
        else:
            img_noise = np.asarray(img)

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img_noise)
        # plt.show()

        img_noise_pil = Image.fromarray(img_noise.astype("uint8"))

        ssim = compare_ssim(img, img_noise_pil, GPU=False)
        label = np.clip(ssim, 0, 1).astype("float32")  # sometimes get ssim slightly larger than 1
        # print(label)

        img_noise = np.transpose(img_noise, [2, 0, 1]).astype("float32")
        return (img_noise - 255 / 2) / 255, label

    def __len__(self):
        return self.length


class MAEMetric(paddle.metric.Metric):
    def __init__(self):
        super(MAEMetric, self).__init__()
        self.absolute_error = 0
        self.batch_count = 0

    def name(self):
        return "MAE"

    def update(self, preds, labels):
        # print(type(preds), type(labels))
        # print(preds, labels)
        self.absolute_error += np.abs(preds - labels).mean()
        self.batch_count += 1

    def accumulate(self):
        return self.absolute_error / self.batch_count

    def reset(self):
        self.absolute_error = 0
        self.batch_count = 0


def train(dataset_path, batch_size=32):
    model = paddle.Model(resnet50(pretrained=True, num_classes=1))

    train_dataset = paddle.io.DataLoader(
        Dataset(dataset_path, "train"), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dataset = paddle.io.DataLoader(
        Dataset(dataset_path, "val"), batch_size=batch_size, shuffle=False, drop_last=True
    )

    optimizer = Momentum(
        learning_rate=0.001, momentum=0.9, weight_decay=L2Decay(1e-4), parameters=model.parameters()
    )
    model.prepare(optimizer, paddle.nn.MSELoss(), MAEMetric())
    model.fit(
        train_dataset,
        val_dataset,
        epochs=1,
        eval_freq=1,
        log_freq=1,
        save_dir="./model/ckpt",
        num_workers=8,
    )
    model.save("./model/ssim", False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SSIM prediction model")
    parser.add_argument("--dataset_path", "-d", type=str, default="./data/demo", help="Path to dataset")
    parser.add_argument("--batch_size", "-bs", type=int, default=1, help="The batch size for training and evaluation")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train(**parse_args().__dict__)
