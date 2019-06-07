import numpy as np
import cv2
from keras.preprocessing import image

from multiprocessing import Pool
from time import time

import imageio
import scipy.ndimage
import requests
import json
from PIL import Image, ImageFilter
import PIL.ImageOps
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import urllib3
urllib3.disable_warnings()

#-----------------------------
#opencv initialization
def main():
    timeLastCall = time()

    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
	#-----------------------------
	#face expression recognizer initialization
    from keras.models import model_from_json
    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
    model.load_weights('facial_expression_model_weights.h5') #load weights

	#-----------------------------
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

	# pool = Pool(processes=1)

    # Traitement Image
    imgUrl = "https://i.pinimg.com/736x/ce/52/d3/ce52d32597fbc0fbc60a2696f4012733.jpg"

    start_img = imageio.imread(imgUrl)

    # start_img.shape(196, 160, 30)

    gray_img = grayscale(start_img)

    inverted_img = 255 - gray_img

    blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=100)

    final_img = dodge(blur_img, gray_img)

    plt.imshow(final_img, cmap='gray')

    plt.imsave('img58.jpg', final_img, cmap='gray', vmin=0, vmax=255)

    # Create an image object

    image1 = Image.open("img58.jpg")

    # Find the edges by applying the filter ImageFilter.FIND_EDGES

    imageWithEdges = (image1.filter(ImageFilter.EDGE_ENHANCE_MORE)).filter(ImageFilter.FIND_EDGES)

    # display the original show
    inverted_image = PIL.ImageOps.invert(imageWithEdges)

    sharpened = (image1.filter(ImageFilter.SMOOTH_MORE)).filter(ImageFilter.DETAIL).filter(ImageFilter.CONTOUR)

    # display the new image with edge detection done

    sharpened.save('./img/new_name.png')
    hin = Image.new('RGBA', sharpened.size, (255, 0, 0, 0))
    hin.save('./img/hint.png', 'PNG')

    while(True):
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
            
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
            
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            
            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
            predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			
			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])
			
            emotion = emotions[max_index]
			
			#write emotion text above rectangle
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			
			#call function every seconds
            if timeLastCall + 10 < time():
                print("Testing ...")
                newImage()
                timeLastCall = time()

			#process on detected face end
			#-------------------------

        cv2.imshow('img',img)

        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break

	#kill open cv things		
    cap.release()
    cv2.destroyAllWindows()

def grayscale(rgb): return np.dot(rgb[..., :3], [0.200, 0.500, 0.114])

def dodge(front, back):
    result = front * 300 / (300 - back)

    result[result > 255] = 255

    result[back == 255] = 255

    return result.astype('uint8')

def newImage():
    URL = "https://dvic.devinci.fr/dgx/paints_torch/api/v1/colorizer"
    hint = open('./img/hint.png', 'rb')
    hint_read = hint.read()
    hint_64_encode = base64.encodebytes(hint_read)

    myImage = open('./img/new_name.png', 'rb')
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

    final_img.save('./img/images.png', 'PNG')

main()