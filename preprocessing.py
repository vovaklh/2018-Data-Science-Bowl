import cv2
import numpy as np


def preprocess(image, height, width, convert=True):
    image = cv2.resize(image, (height, width))

    if convert:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.reshape(height, width, 1)
    image = image / 255

    return image
