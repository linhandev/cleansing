import hashlib
from functools import partial

import numpy as np
import imagehash
from .util import idxs


def md5(img):
    return hashlib.md5(img.data).hexdigest()


hash_size = 16
compute_hash = {
    "md5": md5,
    "ahash": partial(imagehash.average_hash, hash_size=hash_size),
    "phash": partial(imagehash.phash, hash_size=hash_size),
    "dhash": partial(imagehash.dhash, hash_size=hash_size),
    "colorhash": imagehash.colorhash,
    "whash-haar": partial(imagehash.whash, hash_size=hash_size),
    "whash-db4": partial(imagehash.whash, mode="db4", hash_size=hash_size),
}

# TODO: cnn


def cal_distance(digests, key):
    distances = np.zeros([len(digests), len(digests)])
    for ida, idb in idxs(digests):
        distances[ida][idb] = distances[idb][ida] = (
            np.sum((digests[ida][key] - digests[idb][key]) ** 2) ** 0.5
        )
    return distances


def cal_distance_np(digests):
    distances = np.zeros([len(digests), len(digests)])
    for ida, idb in idxs(digests):
        distances[ida][idb] = distances[idb][ida] = (
            np.sum((digests[ida] - digests[idb]) ** 2) ** 0.5
        )
    return distances
