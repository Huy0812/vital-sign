import cv2
import numpy as np
import time
def raw_bvp(image, mask, option=1):
    mean_image = cv2.mean(image, mask)
    mean_red = mean_image[0]
    mean_green = mean_image[1]
    mean_blue = mean_image[2]

    if option == 1:
        return mean_green
    else:
        std_image = cv2.meanStdDev(image, mask=mask)
        std_red = float(std_image[0][0])
        std_blue = float(std_image[0][2])
        spo2 = 1 - 0.04*((std_red/mean_red)/(std_blue/mean_blue))
        #spo2 = 0
        return mean_green, spo2
