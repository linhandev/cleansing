from hmac import digest


from collections import defaultdict
from turtle import distance

import numpy as np


def ids(int_or_collection):
    if not isinstance(int_or_collection, int):
        total_number = len(int_or_collection)
    else:
        total_number = int_or_collection
    for ida in range(total_number):
        for idb in range(ida):
            yield ida, idb


def cal_distance(digests, key):
    distances = np.ones([len(digests), len(digests)]) * len(digests[0][key]) + 1
    for ida, idb in ids(digests):
        distances[ida][idb] = digests[ida][key] - digests[idb][key]
    return distances


def strict(digests, key="md5"):
    res = []
    for ida, idb in ids(digests):
        if digests[ida][key] == digests[idb][key]:
            res.append((ida, idb))
    return res


def percent(digests, key="ahash", percentage=0.1):
    remove_num = int(len(digests) * len(digests) / 2)
    if remove_num < 1:
        raise ValueError(
            f"Percentage {percentage} specified for deduplicate.percentage result in less than 1 image to exclude"
        )

    distances = cal_distance(digests, key)
    order = np.argsort(distances, axis=None)
    similar = order[:remove_num]
    res = np.unravel_index(similar, distances.shape)
    res = [(ida, idb) for ida, idb in zip(res[0], res[1])]
    return res


def thresh(digests, key="ahash", thresh=0.3):
    distances = cal_distance(digests, key) / len(digests[0][key])
    res = []
    for ida, idb in ids(digests):
        if distances[ida][idb] < thresh:
            res.append((ida, idb))
    return res
