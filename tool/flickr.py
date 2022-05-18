import time
import flickrapi
import skimage.io
import skimage.transform
import requests
from io import BytesIO
from tqdm import tqdm
import os
import os.path as osp
import xml.etree.ElementTree as ET

api_key = "d4be5b6e28bcec5f10e61dac50103d0d"
api_secret = "888d638c47509b6d"
flickr = flickrapi.FlickrAPI(api_key, api_secret)
total = 0
kw_total = 0
i = 0

#
for keyword in [
    "bicycle",
    "car",
    "people",
    "shoe",
    "cat",
    "dog",
    "children",
    "boat",
    "mountain",
    "sky",
    "gpu",
    "laptop",
    "monitor",
    "storm",
    "supermoon",
    "park",
    "yellowstone",
    "fish",
    "nature",
    "party",
    "flower",
]:
    total += kw_total
    kw_total = 0
    kw_total += i
    print(keyword, ", keyword total", kw_total, ", dataset total", total)
    i = 0

    save_dir = osp.join("data", keyword)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for page_idx in range(10):
        print("page_idx", page_idx)
        page = flickr.photos.search(
            text=keyword,
            tag_mode="all",
            tags=keyword + ",HD",
            extras="url_c",
            sort="relevance",
            per_page=500,
            page=page_idx,
            content_type=1,
        )

        import warnings

        for photo in tqdm(page.findall("./photos/photo")):
            tic = time.time()

            size_res = flickr.photos.getSizes(photo_id=photo.get("id"))
            sizes = []
            for size in size_res.findall("./sizes/size"):
                try:
                    sizes.append((int(size.get("height")), size.get("source")))
                except:
                    pass
            sizes.sort(key=lambda s: s[0], reverse=True)
            if sizes[0][0] < 1024:
                continue
            # print([s[0] for s in sizes])
            fav_res = flickr.photos.getFavorites(photo_id=photo.get("id"))
            # print(ET.tostring(fav_res))
            # print(fav_res.find("photo").get('total'))
            if fav_res.find("photo").get("total") == 0:
                continue

            for size in sizes:
                if size[0] < 1024:
                    url = size[1]
                    break

            if url is not None:
                try:
                    response = requests.get(url)
                    file = BytesIO(response.content)

                    # Read image from file

                    img = skimage.io.imread(file)

                    # Resize images
                    s = img.shape
                    if len(s) != 3:
                        continue

                    img = skimage.transform.resize(
                        img, (512, 512), order=2, mode="constant", anti_aliasing=True
                    )

                    # Convert to uint8, suppress the warning about the precision loss
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        img = skimage.img_as_ubyte(img)

                    save_path = osp.join(save_dir, f"{keyword}_{str(i).zfill(5)}.png")

                    skimage.io.imsave(save_path, img)

                    i = i + 1
                except Exception as e:
                    print("error", e)
            
            time.sleep(max(2 - (time.time() - tic), 0)) # download at most 3k6/h
