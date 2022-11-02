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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = RoIExtraction()
        self.raw_bvp_arr_forehead = []
        self.raw_bvp_arr_nose = []
        self.raw_bvp_arr_face = []
        self.frame_arr = []
        self.time = []
        self.ibi = []
        self.spo2_face = []
        self.fps = 0
        self.rmssd = 0
        self.heartrate = []
        self.mean_previous = None
        self.grabbed = None
        self.time_run = 60
        self.count = 0

    def __call__(self):
        # self.pipe = pipe
        try:
            thread1 = threading.Thread(target=self.capture_frames)
            thread2 = threading.Thread(target=self.mask_process)
            thread3 = threading.Thread(target=self.vital_sign)
            thread1.start()
            thread2.start()
            thread3.start()
            ani = FuncAnimation(plt.gcf(), self.animate, interval=1000)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(e)

    def animate(self, i):
        data = self.raw_bvp_arr_nose
        x_vals = range(len(data))
        y_vals = data
        plt.cla()
        if (len(self.time) > 0) :
            if self.time[-1] - self.time[0] > self.time_run :
                sys.exit()
        if (len(x_vals) == len(y_vals)):
            plt.plot(x_vals, y_vals, label="signal")
            plt.legend(loc='upper left')
            plt.tight_layout()

    def capture_frames(self, source=0):
        camera = cv2.VideoCapture(source)
        (self.grabbed, frame) = camera.read()
        time_begin = time.time()
        self.frames_count = 0

        while self.grabbed:
            time.sleep(1 / 1000000)
            (grabbed, frame) = camera.read()

            if not grabbed:
                continue

            k = cv2.waitKey(1)

            if k != -1:
                self.terminate(camera)
                break

            (check_y_channel, mean_y_channel) = is_Y_channel(frame, self.mean_previous)
            # frame not satisfy luminance condition
            if (is_low_contrast(frame) or not check_y_channel):
                self.time.clear()
                self.raw_bvp_arr_face.clear()
                self.raw_bvp_arr_forehead.clear()
                self.raw_bvp_arr_nose.clear()

            else:
                (check_luminance, orig) = mean_grayscale(frame)

                if (not (check_luminance)):
                    continue

                else:
                    self.frame_arr.append(frame)

            self.mean_previous = mean_y_channel
            self.frames_count += 1

            if self.frames_count % 30 == 29:
                time_end = time.time()
                self.fps = 30 / (time_end - time_begin)
                sys.stdout.write(f'\rFPS: {self.fps}')
                sys.stdout.flush()
                time_begin = time.time()
            if (len(self.time) > 0) :
                if self.time[-1] - self.time[0] > self.time_run :
                    sys.exit()
        self.terminate(camera)
    
    def mask_process(self):

        while (True):
            time.sleep(1 / 1000000)
            if (len(self.frame_arr) > 0):
                frame = self.frame_arr.pop()
                self.mask(frame)
                forehead_roi = np.asarray(self.mask.getForehead_roi(), dtype="uint8")
                nose_roi = np.asarray(self.mask.getNose_roi(), dtype="uint8")
                face_roi = np.asarray(self.mask.getFace_roi(), dtype="uint8")
                frame_ROI = cv2.drawContours(frame, [np.array(self.mask.outline_forehead)], 0, (255, 0, 0), 2)
                frame_ROI = cv2.drawContours(frame, [np.array(self.mask.outline_nose)], 0, (0, 255, 0), 2)
                frame_ROI = cv2.drawContours(frame, [np.array(self.mask.outline_face)], 0, (0, 0, 255), 2)
                # frame_ROI = cv2.drawContours(frame, [np.array(self.mask.outline_rule)], 0, (0, 255, 255), 2)
                # frame_ROI = cv2.drawContours(frame, [np.array(self.mask.outline_rarule)], 0, (255, 0, 255), 2)
                # frame_ROI = cv2.drawContours(frame, [np.array(self.mask.outline_rure)], 0, (0, 255, 255), 2)
                # frame_ROI = cv2.drawContours(frame, [np.array(self.mask.outline_rarure)], 0, (255, 0, 255), 2)
                cv2.imshow('img', frame_ROI)
                cv2.waitKey(1)
                # Raw BVP from green channel

                forehead_BVP = raw_bvp(frame, forehead_roi)
                nose_BVP = raw_bvp(frame, nose_roi)
                face_BVP, spo2_face = raw_bvp(frame, face_roi, 2)
                self.raw_bvp_arr_forehead.append(forehead_BVP)
                self.raw_bvp_arr_nose.append(nose_BVP)
                self.raw_bvp_arr_face.append(face_BVP)
                self.spo2_face.append(spo2_face)
                self.time.append(time.time())
                #print(len(self.spo2_face))
                # print(self.spo2_forehead)
            
            if (len(self.time) > 0) :
                if self.time[-1] - self.time[0] > self.time_run :
                    sys.exit()
            else:
                pass

    def vital_sign(self):

        while (True):
            time.sleep(1 / 1000000)
            if (len(self.time) > 0) :
                if self.time[-1] - self.time[0] > self.time_run :
                    arr = []
                    for i in range (0, len(self.heartrate) - 1) :
                        
                        if abs(self.heartrate[i+1] - self.heartrate[i]) < 2 : 
                            arr.clear()
                            arr.append(self.heartrate[i])
                            arr.append(self.heartrate[i+1])
                    frame = self.frame_arr.pop()
                    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = frame
                    self.mask.DCES(gray_frame)
                    rule_roi = np.asarray(self.mask.getRule_roi(), dtype="uint8")
                    rarule_roi = np.asarray(self.mask.getRarule_roi(), dtype="uint8")
                    rure_roi = np.asarray(self.mask.getRure_roi(), dtype="uint8")
                    rarure_roi = np.asarray(self.mask.getrarure_roi(), dtype="uint8")
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    DCLES = cv2.mean(gray_frame, rule_roi)[0] - cv2.mean(gray_frame, rarule_roi)[0]
                    DCRES = cv2.mean(gray_frame, rure_roi)[0] - cv2.mean(gray_frame, rarure_roi)[0]
                    DCES = (DCLES + DCRES) / 2 
                    frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rule)], 0, (0, 255, 255), 2)
                    frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rarule)], 0, (255, 0, 255), 2)
                    frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rure)], 0, (0, 255, 255), 2)
                    frame_ROI = cv2.drawContours(gray_frame, [np.array(self.mask.outline_rarure)], 0, (255, 0, 255), 2)
                    cv2.imshow('img', frame_ROI)
                    if DCES >= 0 :
                        print("Dark circle under eyes", 100)
                    if DCES < 0 :
                        print("Dark circle under eyes", DCES)
                        print("heart rate: ", mean(arr))
                        print("hrv", self.rmssd * 1000)
                        print("spo2", mean(self.spo2_face))
                    sys.exit()
            if len(self.raw_bvp_arr_forehead) % 101 == 100:
                real_time = self.time[-1] - self.time[len(self.time) - 100]
                signal_forehead = Signal()
                signal_nose = Signal()
                signal_face = Signal()
                signal_forehead.signal = self.raw_bvp_arr_forehead[-100:]
                signal_nose.signal = self.raw_bvp_arr_nose[-100:]
                signal_face.signal = self.raw_bvp_arr_face[-100:]

                # HRV calculation
                peaks_time = []
                j = y = 0
                peaks, _ = find_peaks(signal_face.signal, height=0)
                for peak in peaks:
                    empty = peak / self.fps
                    peaks_time.append(empty)
                for i in range(1, len(peaks_time)):
                    t = peaks_time[i] - peaks_time[i - 1]
                    self.ibi.append(t)
                n = len(self.ibi)
                while j < n - 1:
                    x = (self.ibi[j] - self.ibi[j + 1]) ** 2
                    y += x
                    j += 1
                self.rmssd = sqrt(y / (n - 1))

                signal_forehead(real_time)
                signal_nose(real_time)
                signal_face(real_time)

                power_forehead, freqs_forehead = bandpass_filter(signal_forehead.signal, real_time)
                power_nose, freqs_nose = bandpass_filter(signal_nose.signal, real_time)
                power_face, freqs_face = bandpass_filter(signal_face.signal, real_time)
                power_selection, bpm_selection, max_index = selection_signal(power_forehead, freqs_forehead,
                                                                             power_nose, freqs_nose, power_face,
                                                                             freqs_face)
                self.heartrate.append(60 * bpm_selection[max_index])
                #print("heartrate", self.heartrate)
                
                #print("spo2", mean(self.spo2_face))
                with open("data.txt", 'a') as file:
                    file.write(str(60 * bpm_selection[max_index]) + "\n")

            else:
                pass

    def terminate(self, camera):
        # self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()
