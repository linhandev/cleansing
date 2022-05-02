from re import S
from pathlib import Path

from PIL import Image
import numpy as np
import imagehash

from cleansing.image.util import listdir, to_binary
from cleansing.image.hash import md5
from cleansing.image.deduplicate import similar_pairs


img_dir = Path("cleansing/image/data")
img_paths = listdir(img_dir)
print(f"Found {len(img_paths)} images")
print(img_paths)

digests = []

for img_path in img_paths:
    img = Image.open(str(img_dir / img_path))
    img_np = np.array(img)  # TODO: more efficient way
    img_np.flags.writeable = False

    digest = {
        "md5": imagehash.ImageHash(to_binary(md5(img_np))),
        # "md5": imagehash.ImageHash.from_str(md5(img_np)),
        "ahash": imagehash.average_hash(img),
        "phash": imagehash.phash(img),
        "dhash": imagehash.dhash(img),
        "colorhash": imagehash.colorhash(img),
        "whash-haar": imagehash.whash(img),
        "whash-db4": imagehash.whash(img, mode="db4"),
    }
    digests.append(digest)

# for key, value in digests[0].items():
#     print(key, value)

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
       