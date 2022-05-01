from hmac import digest


from collections import defaultdict
from turtle import distance

import numpy as np


def strict(digests, key="md5"):
    dig_dict = defaultdict(lambda: [])
    for idx, digest in enumerate(digests):
        dig_dict[digest[key]].append(idx)
    res = []
    for paths in dig_dict.values():
        if len(paths) != 1:
            res.append(paths)
    return res


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


def percent(digests, key="ahash", percentage=0.5):
    if len(digests) * percentage < 1:
        raise ValueError(
            f"Percentage {percentage} specified for deduplicate.percentage result in less than 1 image to exclude"
        )

    distances = cal_distance(digests, key)
    order = np.argsort(distances, axis=None)
    print(int(len(digests) * percentage))
    similar = order[: int(len(digests) * percentage)]
    res = np.unravel_index(similar, distances.shape)
    res = [[ida, idb] for ida, idb in zip(res[0], res[1])]
    return res


def thresh(digests, key="ahash", thresh=0.05):
    distances = cal_distance(digests, key) / len(digests[0][key])
    res = []
    for ida, idb in ids(digests):
        if distances[ida][idb] < thresh:
            res.append([ida, idb])
    return res
