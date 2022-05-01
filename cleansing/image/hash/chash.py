import hashlib

def md5(img):
    return hashlib.md5(img.data).hexdigest()