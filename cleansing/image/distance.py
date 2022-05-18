from pathlib import Path
import random

import sklearn.decomposition
import sklearn.cluster
from PIL import Image
import numpy as np

from .util import listdir, idxs
from .hash import compute_hash, cal_distance, cal_distance_np


def get_digest(dataset_path, hashes=["phash"], hash_weights=[1], pca_thresh=0.9, cluster_number=1):
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
            curr_pos += hash.size
    # digests_comb /= digests_comb.std(axis=(0))[None, :]

    # 4. reduce dimension with pca
    if pca_thresh != 1:
        pca = sklearn.decomposition.PCA(n_components=pca_thresh, svd_solver="full")
        pca.fit(digests_comb)
        digests_comb = pca.transform(digests_comb)

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

    max_distance = 0
    for hash_idx, hash_name in enumerate(hashes):
        max_distance += (norm_weights[hash_idx] * hash_weights[hash_idx]) ** 2 * digests[0][
            hash_name
        ].hash.size
    def cluster_digest():
        for digests, img_paths in zip(digests_cluster, img_paths_cluster):
            yield digests, img_paths 

    return cluster_digest(), max_distance


def get_distance(
    dataset_path, hashes=["phash"], hash_weights=[1], pca_thresh=0.9, cluster_number=1
):
    cluster_digest, max_distance = get_digest(
        dataset_path, hashes, hash_weights, pca_thresh, cluster_number
    )

    def cluster_distance():
        for digests, img_paths in cluster_digest:
            yield cal_distance_np(digests), img_paths

    return cluster_distance(), max_distance
