from cmath import e
from http.client import LENGTH_REQUIRED
from re import M
from statistics import mean
import cv2
import numpy as np
import torch
import time
import sys
import matplotlib.pyplot as plt
import threading
from matplotlib import animation
from multiprocessing import Process
from raw_bvp import raw_bvp
from roi_extraction import RoIExtraction
from luminance import mean_grayscale, is_low_contrast, is_Y_channel
from roi_selection import bandpass_filter, selection_signal
from signal_processing import Signal
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from math import sqrt

class CaptureFrames():

    def __init__(self):
        self.frames_count = 0
        self.mask = RoIExtraction()
        self.raw_bvp_arr_forehead = []
        self.raw_bvp_arr_nose = []
        self.raw_bvp_arr_face = []
        self.frame_arr = []
        self.time = []
        self.ibi = []
        self.spo2_face = []
        self.darkCircle = 100
        self.fps = 0
        self.rmssd = 0
        self.heartrate = []
        self.mean_previous = None
        self.grabbed = None
        self.time_run = 60
        self.count = 0
    def dark_circle(self, rule_roi, rarule_roi, rure_roi, rarure_roi):
            DCLES = rule_roi.get_mean_grayscale() - rarule_roi.get_mean_grayscale()
            DCRES = rure_roi.get_mean_grayscale() - rarure_roi.get_mean_grayscale()
            DCES = (DCLES + DCRES) / 2 
            if DCES >= 0 :
                self.darkCircle = 100
            if DCES < 0 :
                self.darkCircle = 100 + DCES
    def vital_sign(self):

        # while (True):
        #     time.sleep(1 / 1000000)
            # if (len(self.time) > 0) :
            #     if self.time[-1] - self.time[0] > self.time_run :
            #         arr = []
            #         for i in range (0, len(self.heartrate) - 1) :
                        
            #             if abs(self.heartrate[i+1] - self.heartrate[i]) < 2 : 
            #                 arr.clear()
            #                 arr.append(self.heartrate[i])
            #                 arr.append(self.heartrate[i+1])
                    # frame = self.frame_arr.pop()
                    # #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # gray_frame = frame
                    # self.mask.DCES(gray_frame)
                    # rule_roi = np.asarray(self.mask.getRule_roi(), dtype="uint8")
                    # rarule_roi = np.asarray(self.mask.getRarule_roi(), dtype="uint8")
                    # rure_roi = np.asarray(self.mask.getRure_roi(), dtype="uint8")
                    # rarure_roi = np.asarray(self.mask.getrarure_roi(), dtype="uint8")
                    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # DCLES = cv2.mean(gray_frame, rule_roi)[0] - cv2.mean(gray_frame, rarule_roi)[0]
                    # DCRES = cv2.mean(gray_frame, rure_roi)[0] - cv2.mean(gray_frame, rarure_roi)[0]
                    # DCES = (DCLES + DCRES) / 2 
                    # frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rule)], 0, (0, 255, 255), 2)
                    # frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rarule)], 0, (255, 0, 255), 2)
                    # frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rure)], 0, (0, 255, 255), 2)
                    # frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rarure)], 0, (255, 0, 255), 2)
                    # cv2.imshow('img', frame_ROI)
                    # if DCES >= 0 :
                    #     print("Dark circle under eyes", 100)
                    # if DCES < 0 :
                    #    print("Dark circle under eyes", DCES)
                    #print("heart rate: ", mean(arr))
                    #print("hrv", self.rmssd * 1000)
                    #print("spo2", mean(self.spo2_face))
                    #sys.exit()
        if len(self.raw_bvp_arr_forehead) % 2 == 1 and len(self.raw_bvp_arr_forehead) > 600:
            signal_forehead = Signal()
            #signal_nose = Signal()
            #signal_face = Signal()
            signal_forehead.signal = self.raw_bvp_arr_forehead[-600:]
            #signal_nose.signal = self.raw_bvp_arr_nose[-100:]
            #signal_face.signal = self.raw_bvp_arr_face[-100:]

            # HRV calculation
            # peaks_time = []
            # j = y = 0
            # peaks, _ = find_peaks(signal_face.signal, height=0)
            # for peak in peaks:
            #     empty = peak / self.fps
            #     peaks_time.append(empty)
            # for i in range(1, len(peaks_time)):
            #     t = peaks_time[i] - peaks_time[i - 1]
            #     self.ibi.append(t)
            # n = len(self.ibi)
            # while j < n - 1:
            #     x = (self.ibi[j] - self.ibi[j + 1]) ** 2
            #     y += x
            #     j += 1
            # self.rmssd = sqrt(y / (n - 1))

            signal_forehead(39)
            power_forehead, freqs_forehead = bandpass_filter(signal_forehead.signal, 39)
            print("-------------------", len(signal_forehead.signal))
            print(freqs_forehead)
            max_index_forehead = np.argmax(power_forehead)
            self.heartrate.append(60 * freqs_forehead[max_index_forehead])
                
                #power_nose, freqs_nose = bandpass_filter(signal_nose.signal, real_time)
                #power_face, freqs_face = bandpass_filter(signal_face.signal, real_time)
                # power_selection, bpm_selection, max_index = selection_signal(power_forehead, freqs_forehead,
                #                                                              power_nose, freqs_nose, power_face,
                #                                                              freqs_face)
                # self.heartrate.append(60 * bpm_selection[max_index])
                #print("heartrate", self.heartrate)
                
                #print("spo2", mean(self.spo2_face))
            with open("data.txt", 'a') as file:
                file.write(str(60 * freqs_forehead[max_index_forehead]) + "\n")
