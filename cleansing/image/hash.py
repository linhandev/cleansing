import hashlib
from functools import partial

import numpy as np
import imagehash


def md5(img):
    return hashlib.md5(img.data).hexdigest()


compute_hash = {
    "md5": md5,
    "ahash": imagehash.average_hash,
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "colorhash": imagehash.colorhash,
    "whash-haar": imagehash.whash,
    "whash-db4": partial(imagehash.whash, mode="db4"),
}

# TODO: cnn


def ids(int_or_collection):
    if not isinstance(int_or_collection, int):
        total_number = len(int_or_collection)
    else:
        total_number = int_or_collection
    for ida in range(total_number):
        for idb in range(ida):
            yield ida, idb


def cal_distance(digests, key):
    distances = np.zeros([len(digests), len(digests)])
    for ida, idb in ids(digests):
        dis = digests[ida][key] - digests[idb][key]
        distances[ida][idb] = distances[idb][ida] = dis 
    return distances
