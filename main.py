import numpy as np
import cv2
from keras.preprocessing import image
from time import time
import imageio
import scipy.ndimage
import requests
import json
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import urllib3

urllib3.disable_warnings()


# -----------------------------
# opencv initialization
def main():
    timeLastCall = time()

    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    # -----------------------------
    # face expression recognizer initialization
    from keras.models import model_from_json
    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
    model.load_weights('facial_expression_model_weights.h5')  # load weights

    # -----------------------------
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    # pool = Pool(processes=1)

    # Traitement Image
    imgUrl = "http://www.michellart.com/Images/ArcimboldoAutumn1573_Diapo.jpg"

    start_img = imageio.imread(imgUrl)

    # start_img.shape(196, 160, 30)

    gray_img = grayscale(start_img)

    inverted_img = 255 - gray_img

    blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=40)

    final_img = dodge(blur_img, gray_img)

    plt.imshow(final_img, cmap='gray')

    plt.imsave('./img/new_image.png', final_img, cmap='gray', vmin=0, vmax=255)

    # Create an image object

    image1 = Image.open("./img/new_image.png")

    # display the new image with edge detection done

    while (True):
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image

            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            predictions = model.predict(img_pixels)  # store probabilities of 7 expressions

            # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])

            emotion = emotions[max_index]

            # write emotion text above rectangle
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # call function every seconds
            if timeLastCall + 10 < time():
                newImage(emotion, image1.size)
                #"./ img / images.png"

                new_image = cv2.imread('./img/images.png', 1)

                cv2.imshow('image', new_image)

                timeLastCall = time()

            # process on detected face end
            # -------------------------

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


def grayscale(rgb): return np.dot(rgb[..., :3], [0.299, 0.587, 0.300])


def dodge(front, back):
    result = front * 300 / (300 - back)

    result[result > 255] = 255

    result[back == 255] = 255

    return result.astype('uint8')



def newImage(emotion, size):

    URL = "https://dvic.devinci.fr/dgx/paints_torch/api/v1/colorizer"

    myImage = open('./img/new_image.png', 'rb')
    result_read = myImage.read()
    result_64_encode = base64.encodebytes(result_read)

    newHint(size, emotion)

    hint = open('./img/hint.png', 'rb')
    hint_read = hint.read()
    hint_64_encode = base64.encodebytes(hint_read)

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

    final_img.save('./img/images.png', 'PNG')

    return final_img
    #final_img.show()


def newHint(size, emotion):
    hin = Image.new('RGBA', size, (255, 0, 0, 0))

    if 'angry' == emotion:

        main_color = (165, 31, 24)
        second_color = (255, 19, 7)
        third_color = (255, 97, 7)

    elif 'disgust' == emotion:

        main_color = (12, 147, 7)
        second_color = (14, 119, 77)
        third_color = (102, 130, 32)

    elif 'fear' == emotion:

        main_color = (114, 162, 181)
        second_color = (69, 98, 109)
        third_color = (121, 136, 142)

    elif 'happy' == emotion:

        main_color = (216, 45, 116)
        second_color = (255, 104, 167)
        third_color = (255, 183, 213)

    elif 'sad' == emotion:

        main_color = (132, 117, 145)
        second_color = (145, 117, 130)
        third_color = (124, 127, 127)

    elif 'surprise' == emotion:

        main_color = (237, 221, 104)
        second_color = (255, 230, 50)
        third_color = (249, 236, 139)

    else:
        hin.save('./img/hint.png', 'PNG')
        return

    draw = ImageDraw.Draw(hin)
    draw.line((size[0] / 3, size[1] / 4 + 4, size[0] / 3, size[1] / 4), fill=main_color, width=3)
    draw.line((size[0] / 3 * 2, size[1] / 4 + 4, size[0] / 3 * 2, size[1] / 4), fill=main_color, width=3)
    draw.line((size[0] / 3, size[1] / 4 * 2 + 4, size[0] / 3, size[1] / 4 * 2), fill=second_color, width=3)
    draw.line((size[0] / 3 * 2, size[1] / 4 * 2 + 4, size[0] / 3 * 2, size[1] / 4 * 2), fill=second_color, width=3)
    draw.line((size[0] / 3, size[1] / 4 * 3 + 4, size[0] / 3, size[1] / 4 * 3), fill=third_color, width=3)
    draw.line((size[0] / 3 * 2, size[1] / 4 * 3 + 4, size[0] / 3 * 2, size[1] / 4 * 3), fill=third_color, width=3)

    hin.save('./img/hint.png', 'PNG')

    return

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

main()
