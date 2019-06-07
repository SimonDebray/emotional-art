import numpy as np
import cv2
from keras.preprocessing import image
from time import time
import imageio
import scipy.ndimage
import requests
import json
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import urllib3

urllib3.disable_warnings()


# -----------------------------
# open cv initialization
def main():
    time_last_call = time()

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
    img_url = "https://fineartamerica.com/images/artworkimages/medium/1/on-the-way-to-the-dance--celebrating-cinco-de-mayo-karla-horst.jpg"
    # img_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHfk89RY0vfgSmIWHVUNblxb9jwMzD1hf4WgtzNfRrUywzvWSs"
    # img_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_sPl3nRo5bvTX85PYTr96EV9rsrmyoDEFGRUSDgprXoQnSXAi"

    start_img = imageio.imread(img_url)

    # start_img.shape(196, 160, 30)

    gray_img = greyscale(start_img)

    inverted_img = 255 - gray_img

    blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=40)

    final_img = dodge(blur_img, gray_img)

    plt.imshow(final_img, cmap='gray')

    plt.imsave('./img/new_image.png', final_img, cmap='gray', vmin=0, vmax=255)

    # Create an image object

    image1 = Image.open("./img/new_image.png")

    # Find the edges by applying the filter ImageFilter.FIND_EDGES

    # image_with_edges = (image1.filter(ImageFilter.EDGE_ENHANCE_MORE)).filter(ImageFilter.FIND_EDGES)

    # display the original show
    # inverted_image = PIL.ImageOps.invert(image_with_edges)

    sharpened = (image1.filter(ImageFilter.SMOOTH_MORE)).filter(ImageFilter.DETAIL).filter(ImageFilter.CONTOUR)

    image1 = sharpened

    image1.save("./img/new_image.png")

    # display the new image with edge detection done

    while True:
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

            emotion = emotions[int(max_index)]

            # write emotion text above rectangle
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # call function every seconds
            if time_last_call + 10 < time():
                new_image(emotion, image1.size)
                # "./ img / images.png"

                display_image = cv2.imread('./img/images.png', 1)

                cv2.imshow('image', display_image)

                time_last_call = time()

            # process on detected face end
            # -------------------------

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


def greyscale(rgb): return np.dot(rgb[..., :3], [0.299, 0.587, 0.300])


def dodge(front, back):
    result = front * 300 / (300 - back)

    result[result > 255] = 255

    result[back == 255] = 255

    return result.astype('uint8')


def new_image(emotion, size):
    url = "https://dvic.devinci.fr/dgx/paints_torch/api/v1/colorizer"

    my_image = open('./img/new_image.png', 'rb')
    result_read = my_image.read()
    result_64_encode = base64.encodebytes(result_read)

    new_hint(size, emotion)

    hint = open('./img/hint.png', 'rb')
    hint_read = hint.read()
    hint_64_encode = base64.encodebytes(hint_read)

    json_data = json.dumps({
        'sketch': result_64_encode.decode("utf-8"),
        'hint': hint_64_encode.decode("utf-8"),
        'opacity': 0
    })

    headers = {'Content-type': 'application/json; charset=utf-8', 'dataType': 'json'}

    # sending get request and saving the response as response object
    r = requests.post(url=url, data=json_data, headers=headers, verify=False)

    r.raise_for_status()

    json_response = r.json()

    img_data = base64.b64decode(json_response['colored'].split(",")[1])

    final_img = Image.open(BytesIO(img_data))

    final_img.save('./img/images.png', 'PNG')

    return final_img


def new_hint(size, emotion):
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
    draw.line((size[0] / 3, size[1] / 4 * 2 + 4, size[0] / 3, size[1] / 4 * 2), fill=second_color, width=2)
    draw.line((size[0] / 3 * 2, size[1] / 4 * 2 + 4, size[0] / 3 * 2, size[1] / 4 * 2), fill=second_color, width=2)
    draw.line((size[0] / 3, size[1] / 4 * 3 + 4, size[0] / 3, size[1] / 4 * 3), fill=third_color, width=2)
    draw.line((size[0] / 3 * 2, size[1] / 4 * 3 + 4, size[0] / 3 * 2, size[1] / 4 * 3), fill=third_color, width=2)

    hin.save('./img/hint.png', 'PNG')

    return


main()
