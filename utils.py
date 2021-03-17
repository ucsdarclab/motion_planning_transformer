''' Common functions used in this library.
'''
import skimage.io
import io


def png_decoder(key, value):
    '''
    PNG decoder with gray images.
    :param key:
    :param value:
    '''
    if not key.endswith(".png"):
        return None
    assert isinstance(value, bytes)
    return skimage.io.imread(io.BytesIO(value), as_gray=True)


def cls_decoder(key, value):
    '''
    Converts class represented as bytes to integers.
    :param key:
    :param value:
    :returns the decoded value
    '''
    if not key.endswith(".cls"):
        return None
    assert isinstance(value, bytes)
    return int(value)

