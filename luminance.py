from statistics import mean
import cv2
import numpy as np
import skimage


# mean_grayscale
# if the range of the mean of all pixel values in grayscale is 127-255 return frame
# if the range of the mean of all pixel values in grayscale is 75-127 skip frame
# if  the range of the mean of all pixel values in grayscale is 0-75 fixed frame with histogram equalizattion

def mean_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_grayImage = gray_img.mean()

    if (mean_grayImage > 127):
        return True, img

    elif (mean_grayImage >= 75 and mean_grayImage <= 127):
        return False, img

    else:
        img_yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_yCrCb[:, :, 0] = cv2.equalizeHist(img_yCrCb[:, :, 0])
        img = cv2.cvtColor(img_yCrCb, cv2.COLOR_YCrCb2BGR)
        return True, img


# low contrast: frame contrast is compared with threshhold of 0.65

def is_low_contrast(img):
    return (skimage.exposure.is_low_contrast(img, 0.35))


# Y channel to YCrCb model: compare the mean value of Y channel of current frame with previous farame > 15

def is_Y_channel(img, mean_previous):
    img_yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = np.array(img_yCrCb[:, :, 0])
    mean_y_channel = y_channel.mean()

    if mean_previous is None:
        return True, mean_y_channel

    else:

        if abs(mean_y_channel - mean_previous) > 15:
            return False, mean_y_channel

        else:
            return True, mean_y_channel
