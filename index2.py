import imageio
import numpy as np
import scipy.ndimage
import requests
import json
from PIL import Image, ImageFilter
import PIL.ImageOps
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def grayscale(rgb): return np.dot(rgb[..., :3], [0.200, 0.500, 0.114])


img = "https://i.pinimg.com/736x/ce/52/d3/ce52d32597fbc0fbc60a2696f4012733.jpg"

start_img = imageio.imread(img)

# start_img.shape(196, 160, 30)

gray_img = grayscale(start_img)

inverted_img = 255 - gray_img

blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=100)


def dodge(front, back):
    result = front * 300 / (300 - back)

    result[result > 255] = 255

    result[back == 255] = 255

    return result.astype('uint8')


final_img = dodge(blur_img, gray_img)

plt.imshow(final_img, cmap='gray')

plt.imsave('img58.jpg', final_img, cmap='gray', vmin=0, vmax=255)

# Create an image object

image = Image.open("img58.jpg")

# Find the edges by applying the filter ImageFilter.FIND_EDGES

imageWithEdges = (image.filter(ImageFilter.EDGE_ENHANCE_MORE)).filter(ImageFilter.FIND_EDGES)

# display the original show
inverted_image = PIL.ImageOps.invert(imageWithEdges)

sharpened = (image.filter(ImageFilter.SMOOTH_MORE)).filter(ImageFilter.DETAIL).filter(ImageFilter.CONTOUR)

# display the new image with edge detection done

sharpened.save('new_name.png')
hin = Image.new('RGBA', sharpened.size, (255, 0, 0, 0))
hin.save('hint.png', 'PNG')

URL = "https://dvic.devinci.fr/dgx/paints_torch/api/v1/colorizer"

# defining a params dict for the parameters to be sent to the API

hint = open('hint.png', 'rb')
hint_read = hint.read()
hint_64_encode = base64.encodebytes(hint_read)

myImage = open('new_name.png', 'rb')
result_read = myImage.read()
result_64_encode = base64.encodebytes(result_read)

jsonData = json.dumps({
    'sketch': result_64_encode.decode("utf-8"),
    'hint': hint_64_encode.decode("utf-8"),
    'opacity': 0
})

headers = {'Content-type': 'application/json; charset=utf-8', 'dataType': 'json'}

# sending get request and saving the response as response object
r = requests.post(url=URL, data=jsonData, headers=headers, verify=False)

data = r.raise_for_status()

jsonResponse = r.json()

imgData = base64.b64decode(jsonResponse['colored'].split(",")[1])

final_img = Image.open(BytesIO(imgData))

final_img.save('images.png', 'PNG')
