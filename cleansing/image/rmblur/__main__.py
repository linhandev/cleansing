import argparse

from .predict import predict


def parse_args(dataset_path, weight_path, thresh):
    parser = argparse.ArgumentParser(description="Predict with SSIM model")
    parser.add_argument("--dataset_path", "-i", type=str, help="Path to image")
    parser.add_argument("--weight_path", "-w", type=str, default="./model/ssim")
    parser.add_argument("--thresh", "-t", type=float)
    args = parser.parse_args()
    return args

def rmblur():




if __name__ == "__main__":
    rmblur(**(parse_args().__dict__))

