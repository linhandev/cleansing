from re import S
from pathlib import Path
import hashlib

from PIL import Image
import numpy as np
import imagehash

from util import listdir, to_binary
from hash import md5
from routine import deduplicate


img_dir = Path("data")
img_paths = listdir(img_dir)
print(f"Found {len(img_paths)} images")
print(img_paths)

digests = []

for img_path in img_paths:
    img = Image.open(str(img_dir / img_path))
    img_np = np.array(img)  # TODO: more efficient way
    img_np.flags.writeable = False

    digest = {
        # "path": img_path,
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
    res = deduplicate.strict(digests, key)
    print(key)
    for idxs in res:
        print([img_paths[idx] for idx in idxs])
        
print("\n\npercent: ")
for key in digests[0].keys():
    res = deduplicate.percent(digests, key)
    print(key)
    for idxs in res:
        print([img_paths[idx] for idx in idxs])
       
print("\n\nthresh: ")
for key in digests[0].keys():
    if key != "colorhash":
        continue
    res = deduplicate.thresh(digests, key)
    print(key)
    for idxs in res:
        print([img_paths[idx] for idx in idxs])
       