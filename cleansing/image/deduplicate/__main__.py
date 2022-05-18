from pathlib import Path
import argparse
import random

from PIL import Image
import numpy as np

from ..util import listdir
from . import similar_pairs
from ..hash import compute_hash, cal_distance

def parse_args():

    parser = argparse.ArgumentParser(
        description="Remove duplicate images from a dataset. Note that thresh and percentage can't be used at the same time!"
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--percentage",
        type=int,
        help="percentage of images to keep. 0.9 percentage on dataset with 100 images leaves 90 after deduplication",
    )
    parser.add_argument(
        "--thresh", type=float, help="0~1, 1 means the same, larger thresh keeps more images"
    )
    parser.add_argument(
        "--partations",
        type=int,
        default=1,
        help="cluster the images before deduplication, images from different clusters won't be compared. Set to 1 if you don't wish to partation dataset.",
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
    unknown_hash = set(args.hashes) - set(
        ["ahash", "phash", "dhash", "colorhash", "whash-haar", "whash-db4", "cnn"]
    )
    assert len(unknown_hash) == 0, f"Hash function {unknown_hash} is not supported"
    return args


def deduplicate(dataset_path, percentage, thresh, hashes=["phash"], hash_weights=[1], partations=1):
    # 1. get all image paths
    img_dir = Path(dataset_path)
    img_paths = listdir(img_dir)
    print(f"Found {len(img_paths)} images under {img_dir}")

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

    # 3. randomly sample 1k images, compute distance, normalize hashes
    if len(digests) > 1000:
        digests_part = random.sample(digests, 1000)
    else:
        digests_part = digests
    dis_part = np.zeros([len(hashes), len(digests_part), len(digests_part)])

    for hash_idx, hash_name in enumerate(hashes):
        dis_part[hash_idx, :, :] = cal_distance(digests_part, hash_name)
        print(dis_part[hash_idx])
    stds = dis_part.std(axis=(1, 2))
    std_sum = np.sum(stds)
    weights = [(std_sum - std)/std_sum for std in stds]
    print(stds, std_sum, weights)
    
    # 4. apply balancing weight and user specified weights to digests, combine together
    # digest_combo = []
    # for digest in digests:



    print("\n\nstrict: ")
    for key in digests[0].keys():
        res = similar_pairs.strict(digests, key)
        print(key)
        for idxs in res:
            print([img_paths[idx] for idx in idxs])

    print("\n\npercent: ")
    for key in digests[0].keys():
        res = similar_pairs.percent(digests, key)
        print(key)
        for idxs in res:
            print([img_paths[idx] for idx in idxs])

    print("\n\nthresh: ")
    for key in digests[0].keys():

        res = similar_pairs.thresh(digests, key)
        print(key)
        for idxs in res:
            print([img_paths[idx] for idx in idxs])


if __name__ == "__main__":
    deduplicate(**parse_args().__dict__)
