from pathlib import Path
import argparse
import random

import sklearn.decomposition
import sklearn.cluster
from PIL import Image
import numpy as np

from ..util import listdir, idxs
from ..hash import compute_hash, cal_distance, cal_distance_np
from ..distance import get_distance


def parse_args():

    parser = argparse.ArgumentParser(
        description="Remove redundant (outlier) images from a dataset."
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--percentage",
        type=int,
        help="percentage of images to keep. 0.9 percentage on dataset with 100 images leaves 90 after deredundant",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        help="0~1, isolation score from isolation forest will be normalized and compared with this threshold",
    )
    parser.add_argument(
        "--hashes",
        type=str,
        nargs="+",
        help="the hashes to use, seperate with space, support ahash, phash, dhash, colorhash, whash-haar, whash-db4 and cnn",
    )
    parser.add_argument(
        "--hash_weights",
        type=float,
        nargs="+",
        help="the hashes to use, seperate with space, support ahash, phash, dhash, colorhash, whash-haar, whash-db4 and cnn",
    )

    args = parser.parse_args()
    if args.hash_weights is not None:
        assert len(args.hashes) == len(
            args.hash_weights
        ), f"Number of hash functions to use ({len(args.hashes)}) doesn't equal number of hash weights ({len(args.hash_weights)})"
    else:
        args.hash_weights = [1] * len(args.hashes)
    unknown_hash = set(args.hashes) - set(
        ["ahash", "phash", "dhash", "colorhash", "whash-haar", "whash-db4", "cnn"]
    )
    assert len(unknown_hash) == 0, f"Hash function {unknown_hash} is not supported"
    return args


def deredundant(
    dataset_path,
    percentage,
    thresh,
    hashes=["phash"],
    hash_weights=[1],
    pca_thresh=0.9,
    cluster_number=1,
):
    to_remove = []
    cluster_info, max_distance = get_distance(
        dataset_path, hashes, hash_weights, pca_thresh, cluster_number
    )

if __name__ == "__main__":
    deredundant(**parse_args().__dict__)
