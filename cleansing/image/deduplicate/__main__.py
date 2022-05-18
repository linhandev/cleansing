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
        description="Remove duplicate images from a dataset. Note that thresh and percentage can't be used at the same time!"
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--percentage",
        type=float,
        help="percentage of images to keep. 0.9 percentage on dataset with 100 images leaves 90 after deduplication",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        help="0~1, 0 means the same, 1 means maximum distance, smaller thresh keeps more images.",
    )
    parser.add_argument(
        "--cluster_number",
        type=int,
        default=1,
        help="cluster the images before deduplication, images from different clusters won't be compared. Set to 1 if you don't wish to partation dataset.",
    )
    parser.add_argument(
        "--hashes",
        type=str,
        nargs="+",
        default=["ahash", "dhash"],
        help="the hashes to use, seperate with space, support ahash, phash, dhash, colorhash, whash-haar, whash-db4 and cnn",
    )
    parser.add_argument(
        "--hash_weights",
        type=float,
        nargs="+",
        help="Weight for the hashes, seperate with space, default to all 1. If specified, must have the same length as hashes",
    )
    parser.add_argument(
        "--pca_thresh",
        type=float,
        default=1,
        help="The percentage of variance pca keeps. Defaults to 0.95",
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


def deduplicate(
    dataset_path,
    percentage,
    thresh,
    hashes=["phash"],
    hash_weights=[1],
    pca_thresh=0.9,
    cluster_number=1,
):
    # in each cluster, deduplicate
    to_remove = []
    cluster_info, max_distance = get_distance(
        dataset_path, hashes, hash_weights, pca_thresh, cluster_number
    )

    if thresh is not None:
        thresh *= max_distance
        for distances, img_paths_cluster in cluster_info:
            to_remove_cluster = []

            idx_pair = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
            for ida, idb in zip(idx_pair[0], idx_pair[1]):
                if ida == idb:
                    continue
                if ida in to_remove_cluster or idb in to_remove_cluster:
                    continue
                if distances[ida][idb] <= thresh:
                    to_remove_cluster.append(idb)

            to_remove_cluster = [img_paths_cluster[idx] for idx in to_remove_cluster]
            to_remove += to_remove_cluster
    else:
        for distances, img_paths_cluster in cluster_info:
            to_remove_number = len(img_paths_cluster) * (1 - percentage)
            idx_pair = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
            to_remove_cluster = []

            for ida, idb in zip(idx_pair[0], idx_pair[1]):
                if ida == idb:
                    continue
                if ida in to_remove_cluster or idb in to_remove_cluster:
                    continue
                to_remove_cluster.append(idb)
                if len(to_remove_cluster) >= to_remove_number:
                    break
            to_remove_cluster = [img_paths_cluster[idx] for idx in to_remove_cluster]
            to_remove += to_remove_cluster
    print("Suggest remove image:")
    for path in to_remove:
        print(path, end=" ")
    print()


if __name__ == "__main__":
    deduplicate(**parse_args().__dict__)
