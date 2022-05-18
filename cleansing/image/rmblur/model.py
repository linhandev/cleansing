import os
import random
import cv2
import skimage
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
import paddle
from paddle.vision.models import resnet50, vgg16, LeNet
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from SSIM_PIL import compare_ssim

import matplotlib

matplotlib.use("TkAgg")

# (N,3,H,W), bgr
class Dataset(paddle.io.Dataset):
    def __init__(self, mode):
        data_dir = "./data/bicycle"
        data_paths = os.listdir(data_dir)
        self.data_paths = [osp.join(data_dir, f) for f in data_paths]

        if mode == "train":
            self.length = len(self.data_paths)
        elif mode == "val":
            self.length = 100
        # print(self.data_paths)

    def __getitem__(self, idx):
        img = Image.open(self.data_paths[idx])  # hwc bgr
        # img_noise = skimage.util.random_noise(np.asarray(img), mode="gaussian")
        ksize = int(random.random() * 10)
        print(np.asarray(img).shape, ksize)
        img_noise = cv2.blur(np.asarray(img), (ksize, ksize))

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(img_noise)
        # plt.show()

        img_noise_pil = Image.fromarray(img_noise.astype("uint8"))

        label = compare_ssim(img, img_noise_pil)
        label = np.clip(label, 0, 1)  # sometimes get ssim slightly larger than 1
        print(label)

        img_noise = np.transpose(img_noise, [2, 0, 1]).astype("float32")
        return img_noise, label

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
        print(type(preds), type(labels))
        self.absolute_error += (preds - labels).mean()

    def accumulate(self):
        # 利用update中积累的成员变量数据进行计算后返回
        return self.absolute_error / self.batch_count

    def reset(self):
        self.absolute_error = 0
        self.batch_count = 0


model = paddle.Model(resnet50(pretrained=False, num_classes=1))

# 使用Cifar10数据集
train_dataset = paddle.io.DataLoader(Dataset("train"), batch_size=8, shuffle=True, drop_last=True)
val_dataset = paddle.io.DataLoader(Dataset("val"), batch_size=8, shuffle=False, drop_last=True)

# 定义优化器
optimizer = Momentum(
    learning_rate=0.01, momentum=0.9, weight_decay=L2Decay(1e-4), parameters=model.parameters()
)
# 进行训练前准备
model.prepare(optimizer, paddle.nn.MSELoss(), MAEMetric())
# 启动训练
model.fit(train_dataset, val_dataset, epochs=50, batch_size=64, save_dir="./output", num_workers=8)
