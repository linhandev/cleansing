import os
import os.path as osp

import numpy as np


def to_binary(b16str):
    return np.array([int(c) for c in bin(int(b16str, 16))[2:].zfill(len(b16str) * 4)])


def listdir(folder, filters={"exclude_prefix": ["."]}):
    """
    list all files satisfying filters under folder and its subfolders

    Args:
        folder (str): the folder to list
        filters (dict, optional): Four lists, include/exclude_prefix/postfix. Include first, satisfying either include, then exclude fail either one gets excluded.

    Returns:
        list: File paths relative to folder, sorted
    """

    files = []
    for root, fdrs, fs in os.walk(folder):
        if osp.basename(root).startswith("."):  # skip all hidden folders
            continue
        for f in fs:
            files.append(osp.normpath(osp.join(root, f)))
    files = [osp.relpath(f, folder) for f in files]
    # TODO: support regx
    include_prefix = filters.get("include_prefix", [])
    include_postfix = filters.get("include_postfix", [])

    def include(path):
        f = osp.basename(path)
        for pref in include_prefix:
            if f[: len(pref)].lower() == pref:
                return True
        for postf in include_postfix:
            if f[-len(postf) :].lower() == postf:
                return True
        return False

    if len(include_prefix) != 0 or len(include_postfix) != 0:
        files = list(filter(include, files))

    exclude_prefix = filters.get("exclude_prefix", [])
    exclude_postfix = filters.get("exclude_postfix", [])

    def exclude(path):
        f = osp.basename(path)
        for pref in exclude_prefix:
            if f[: len(pref)] == pref:
                return False
        for postf in exclude_postfix:
            if f[-len(postf) :] == postf:
                return False
        return True

    files = list(filter(exclude, files))
    files.sort()
    files = [osp.normpath(p) for p in files]
    return files


def idxs(int_or_collection):
    if not isinstance(int_or_collection, int):
        total_number = len(int_or_collection)
    else:
        total_number = int_or_collection
    for ida in range(total_number):
        for idb in range(ida):
            yield ida, idb
