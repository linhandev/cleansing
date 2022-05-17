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

api_key = "d4be5b6e28bcec5f10e61dac50103d0d"
api_secret = "888d638c47509b6d"
flickr = flickrapi.FlickrAPI(api_key, api_secret)

# keyword = "bicycle"
for keyword in ["bicycle"]:
    photos = flickr.walk(
        text=keyword, tag_mode="all", tags=keyword, extras="url_c", sort="relevance", per_page=2000
    )
    # print(photos)

    save_dir = osp.join('data', keyword)
    dir_exists = os.path.isdir(save_dir)
    if not dir_exists:
        os.mkdir(save_dir)
        print("Making directory %s" % save_dir)
    else:
        print("Will store images in directory %s" % save_dir)


    import warnings

    nimage = 1500
    i = 0
    size = 256
    for photo in tqdm(photos):
        url = photo.get("url_c")
        if not (url is None):
            response = requests.get(url)
            file = BytesIO(response.content)

            # Read image from file
            img = skimage.io.imread(file)

            # Resize images

            img = skimage.transform.resize(img, (size, size), mode="constant")

            # Convert to uint8, suppress the warning about the precision loss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im2 = skimage.img_as_ubyte(img)

            # Save the image
            # if i < 1000:
            #     save_dir = f"train/{dir_name}"
            # else:
            #     save_dir = f"test/{dir_name}"
            
            local_name = "{0:s}/{1:s}_{2:04d}.jpg".format(save_dir, i)
            
            skimage.io.imsave(local_name, im2)

            i = i + 1
            
        if i >= nimage:
            break
