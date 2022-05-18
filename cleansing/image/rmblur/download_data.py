from posixpath import dirname
import flickrapi
import numpy as np
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

# "bicycle", "car",
for keyword in [ "people", "shoe", "cat", "dog", "children"]:
# for keyword in ["boat", "mountain", "sky", "lap top", "gpu", "monitor"]:
    print(keyword)
    i = 0

    for page_idx in range(10):
        print(page_idx)
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
        # print(ET.tostring(page))

        save_dir = osp.join("data", keyword)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print("Making directory %s" % save_dir)
        else:
            print("Will store images in directory %s" % save_dir)

        import warnings

        size = 256
        # print(page.findall('./photos/photo'))
        for photo in tqdm(page.findall('./photos/photo')):
            res = flickr.photos.getSizes(photo_id=photo.get("id"))
            sizes = []
            for size in res.findall("./sizes/size"):
                try:
                    sizes.append((int(size.get("height")), size.get("source")))
                except:
                    pass
            sizes.sort(key=lambda s: s[0], reverse=True)
            if sizes[0][0] < 1024:
                continue
            # print([s[0] for s in sizes])
            
            for size in sizes:
                if size[0] < 1024:
                    url = size[1]
                    break
            

            if url is not None:
                response = requests.get(url)
                file = BytesIO(response.content)

                # Read image from file
                img = skimage.io.imread(file)

                # Resize images
                # print(img)
                s = img.shape
                # print(s)
                if len(s) != 3:
                    continue

                img = skimage.transform.resize(
                    img, (512, 512), order=1, mode="constant", anti_aliasing=True
                )

                # Convert to uint8, suppress the warning about the precision loss
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img = skimage.img_as_ubyte(img)

                save_path = osp.join(save_dir, f"{keyword}_{str(i).zfill(5)}.png")

                skimage.io.imsave(save_path, img)

                i = i + 1