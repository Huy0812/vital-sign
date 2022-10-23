import cv2
import numpy as np
def raw_bvp(image, mask, option = 1):
    mean_image = cv2.mean(image, mask)
    mean_red = mean_image[2]
    mean_green = mean_image[1]
    mean_blue = mean_image[0]
    
    if option == 1 :
        return mean_green
    else :
        #test_image = image[mask]
        #std_red = np.std(test_image[2])
        #std_blue = np.std(test_image[0])
        #po2 = 1 - 0.04*((std_red/mean_red)/(std_blue/mean_blue))
        spo2 = 0
        return mean_green, spo2