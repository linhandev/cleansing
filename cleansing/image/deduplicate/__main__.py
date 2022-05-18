from pathlib import Path
import argparse
import random

import sklearn.decomposition
import sklearn.cluster
from PIL import Image
import numpy as np

from ..util import listdir, idxs
from ..hash import compute_hash, cal_distance, cal_distance_np


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
        default=0.99,
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
    # 1. get all image paths
    img_dir = Path(dataset_path)
    img_paths = listdir(img_dir)
    print(f"Found {len(img_paths)} images under {img_dir}")
    if len(img_paths) == 0:
        exit(0)

    # 2. compute all digests
    digests = []
    for img_path in img_paths:
        img = Image.open(str(img_dir / img_path))
        img_np = np.array(img)  # TODO: more efficient way
        img_np.flags.writeable = False

        digest = {}
        for name in hashes:
            digest[name] = compute_hash[name](img)
        digests.append(digest)
    del digest

    # 3. randomly sample 1k images, compute distance, normalize hashes based on std diviation, also apply weight user specify
    if len(digests) > 1000:
        digests_part = random.sample(digests, 1000)
    else:
        digests_part = digests

    dis_part = np.zeros([len(hashes), len(digests_part), len(digests_part)])

    for hash_idx, hash_name in enumerate(hashes):
        dis_part[hash_idx, :, :] = cal_distance(digests_part, hash_name)

    stds = dis_part.std(axis=(1, 2))
    std_sum = np.sum(stds)
    norm_weights = [(std_sum - std) / std_sum / (len(hashes) - 1) for std in stds]
    print(f"Normalizing weights for hashes {hashes} are {norm_weights}")

    total_lengh = 0
    for hash in hashes:
        total_lengh += len(digests[0][hash])

    digests_comb = np.zeros((len(digests), total_lengh))
    for sample_idx in range(len(digests)):
        curr_pos = 0
        for hash_idx, hash_name in enumerate(hashes):
            hash = digests[sample_idx][hash_name].hash
            digests_comb[sample_idx][curr_pos : curr_pos + hash.size] = (
                hash * norm_weights[hash_idx] * hash_weights[hash_idx]
            ).reshape((hash.size))
            # digests_comb[sample_idx][curr_pos : curr_pos + hash.size] = (
            #     hash
            # ).reshape((hash.size))
            curr_pos += hash.size
    # digests_comb /= digests_comb.std(axis=(0))[None, :]

    # 4. reduce dimension with pca
    # pca = sklearn.decomposition.PCA(n_components=pca_thresh, svd_solver="full")
    # pca.fit(digests_comb)
    # digests_comb = pca.transform(digests_comb)

    # 5. partation dataset
    if cluster_number <= 1:
        digests_cluster = [digests_comb]
        img_paths_cluster = [img_paths]
    else:
        kmeans = sklearn.cluster.KMeans(n_clusters=cluster_number, init="k-means++").fit(
            digests_comb
        )
        cluster_ids = kmeans.predict(digests_comb)
        digests_cluster = []
        img_paths_cluster = [[] for _ in range(cluster_number)]
        for cluster_idx in range(cluster_number):
            digests_cluster.append(digests_comb[cluster_ids == cluster_idx])
            for sample_idx, path in enumerate(img_paths):
                if cluster_ids[sample_idx] == cluster_idx:
                    img_paths_cluster[cluster_idx].append(path)
    for idx in range(cluster_number):
        img_paths_cluster[idx].sort()
    print(f"The whole dataset is divided to {cluster_number} parts.")
    for cluster_idx in range(cluster_number):
        print(f"Images in part {cluster_idx}")
        for path in img_paths_cluster[cluster_idx]:
            print(path, end=" ")
        print()

    # 6. in each cluster, deduplicate
    to_remove = []
    if thresh is not None:
        max_distance = 0
        for hash_idx, hash_name in enumerate(hashes):
            max_distance += (norm_weights[hash_idx] * hash_weights[hash_idx]) ** 2 * digests[0][
                hash_name
            ].hash.size
        thresh *= max_distance
        print(6.98919235178537/max_distance)

        for cluster_idx, digests in enumerate(digests_cluster):
            distances = cal_distance_np(digests)
            to_remove_cluster = []

            idx_pair = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
            for ida, idb in zip(idx_pair[0], idx_pair[1]):
                if ida == idb:
                    continue
                if ida in to_remove_cluster or idb in to_remove_cluster:
                    continue
                if ida == 19 and idb ==18:
                    print("+_+_+", distances[ida][idb])
                if distances[ida][idb] <= thresh:
                    print(distances[ida][idb], ida, idb)
                    to_remove_cluster.append(idb)
            
            to_remove_cluster = [img_paths_cluster[cluster_idx][idx] for idx in to_remove_cluster]
            to_remove += to_remove_cluster
    else:
        for cluster_idx, digests in enumerate(digests_cluster):
            to_remove_number = len(img_paths_cluster[cluster_idx]) * (1 - percentage)
            distances = cal_distance_np(digests)
            idx_pair = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
            to_remove_cluster = []
            
            for ida, idb in zip(idx_pair[0], idx_pair[1]):
                if ida == idb:
                    continue
                if ida in to_remove_cluster or idb in to_remove_cluster:
                    continue
                print(distances[ida][idb], ida, idb)
                to_remove_cluster.append(idb)
                if len(to_remove_cluster) >= to_remove_number:
                    break
            to_remove_cluster = [img_paths_cluster[cluster_idx][idx] for idx in to_remove_cluster]
            to_remove += to_remove_cluster
    print("Suggest remove image:")
    for path in to_remove:
        print(path, end=" ")
    print()

if __name__ == "__main__":
    deduplicate(**parse_args().__dict__)
