import argparse

import sklearn.ensemble
import numpy as np

from ..distance import get_digest


def parse_args():

    parser = argparse.ArgumentParser(
        description="Remove redundant (outlier) images from a dataset."
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.1,
        help="percentage of images to keep. 0.9 percentage on dataset with 100 images leaves 90 after deredundant",
    )
    # parser.add_argument(
    #     "--thresh",
    #     type=float,
    #     help="0~1, isolation score from isolation forest will be normalized and compared with this threshold",
    # )
    parser.add_argument(
        "--pca_thresh",
        type=float,
        default=0.95,
        help="The percentage of variance pca keeps. Defaults to 0.95",
    )
    parser.add_argument(
        "--hashes",
        type=str,
        nargs="+",
        default=["ahash", "dhash", "colorhash", "whash-haar"],
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
    hashes=["phash"],
    hash_weights=[1],
    pca_thresh=0.9,
):
    to_remove = []
    digests, max_distance = get_digest(
        dataset_path, hashes, hash_weights, pca_thresh, cluster_number=1
    )

    digests, img_paths = next(digests)

    # print(digests.shape, img_paths)
    clf = sklearn.ensemble.IsolationForest().fit(digests)
    abnormal_scores = clf.score_samples(digests)
    to_remove = np.argsort(abnormal_scores)[: int(percentage * len(abnormal_scores))]
    to_remove = [img_paths[idx] for idx in to_remove]
    print("Suggest remove:")
    for path in to_remove:
        print(path, end=" ")
    print()


if __name__ == "__main__":
    deredundant(**parse_args().__dict__)
