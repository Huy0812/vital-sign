import cv2
import numpy as np
import time
import sys
from raw_bvp import raw_bvp
from roi_extraction import RoIExtraction
from luminance import mean_grayscale, is_low_contrast, is_Y_channel
from roi_selection import bandpass_filter, selection_signal
from signal_processing import Signal


class Process():

    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frames_count = 0
        self.fps = 0
        self.mask = RoIExtraction()
        self.forehead_ROI = []
        self.nose_ROI = []
        self.face_ROI = []
        self.power_forehead = []
        self.freqs_forehead = []
        self.angle = []
        self.bpms = []
        self.mean_previous = None
        self.raw_bvp_arr_forehead = []
        self.raw_bvp_arr_nose = []
        self.raw_bvp_arr_face = []
        self.time_begin_test = None
        self.time_end_test = None
        self.time_test = None

    def run(self):
        time.sleep(1)
        frame = self.frame_in
        time_begin = time.time()

        # long = 0
        frame = self.frame_in
        (check_y_channel, mean_y_channel) = is_Y_channel(frame, self.mean_previous)
        # frame not satisfy luminance condition
        if (is_low_contrast(frame) or not check_y_channel):
            self.time_begin_test = None
            self.raw_bvp_arr_face.clear()
            self.raw_bvp_arr_forehead.clear()
            self.raw_bvp_arr_nose.clear()

        else:
            (check_luminance, orig) = mean_grayscale(frame)
            if (not (check_luminance)):
                cv2.imwrite(f'./removed_images/image{self.frames_count}.png', orig)

            else:
                self.mask(frame)
                self.forehead_ROI = self.mask.outline_forehead
                self.nose_ROI = self.mask.outline_nose
                self.face_ROI = self.mask.outline_face
                # forehead_roi_real = self.mask.getForehead_roi()
                # nose_roi_real =  self.mask.getNose_roi()
                # face_roi_real =  self.mask.getFace_roi()

                forehead_roi = np.asarray(self.mask.getForehead_roi(), dtype="uint8")
                nose_roi = np.asarray(self.mask.getNose_roi(), dtype="uint8")
                face_roi = np.asarray(self.mask.getFace_roi(), dtype="uint8")

                # Raw BVP from green channel

                forehead_BVP = raw_bvp(frame, forehead_roi)
                nose_BVP = raw_bvp(frame, nose_roi)
                face_BVP = raw_bvp(frame, face_roi)
                self.raw_bvp_arr_forehead.append(forehead_BVP)
                self.raw_bvp_arr_nose.append(nose_BVP)
                self.raw_bvp_arr_face.append(face_BVP)
                if (self.time_begin_test == None):
                    self.time_begin_test = time.time()
                self.time_end_test = time.time()

                # with open("data.txt", 'a') as file:
                # file.write(str(len(raw_bvp_arr)) + "\t" + str(face_BVP) + "\n")

                # out_forehead = np.zeros(frame.shape , np.uint8)
                # out_forehead[forehead_roi_real] = frame[forehead_roi_real]
                # out_nose = np.zeros(frame.shape , np.uint8)
                # out_nose[nose_roi_real] = frame[nose_roi_real]
                # out_face = np.zeros(frame.shape , np.uint8)
                # out_face[face_roi_real] = frame[face_roi_real]
                # cv2.imwrite(f'./images/forehead{self.frames_count}.png', out_forehead )
                # cv2.imwrite(f'./images/nose{self.frames_count}.png', out_nose )
                # cv2.imwrite(f'./images/face{self.frames_count}.png', out_face )

        self.mean_previous = mean_y_channel
        if self.frames_count % 30 == 29:
            time_end = time.time()
            self.fps = 30 / (time_end - time_begin)
            time_begin = time.time()
        if (self.time_begin_test != None and self.time_end_test != None):
            if len(self.raw_bvp_arr_forehead) >= 300:
                if (self.time_test == None):
                    self.time_test = self.time_end_test - self.time_begin_test
                # if len(self.raw_bvp_arr_face) == 600 :
                signal_forehead = Signal()
                signal_nose = Signal()
                signal_face = Signal()
                signal_forehead.signal = self.raw_bvp_arr_forehead[-300:]
                signal_nose.signal = self.raw_bvp_arr_nose[-300:]
                signal_face.signal = self.raw_bvp_arr_face[-300:]
                self.angle = signal_forehead.normalize_frequency_signal()

                # fig, axs = plt.subplots(3)
                signal_forehead()
                signal_nose()
                signal_face()
                power_forehead, freqs_forehead = bandpass_filter(signal_forehead.signal, self.time_test)
                self.power_forehead = power_forehead
                self.freqs_forehead = freqs_forehead
                power_nose, freqs_nose = bandpass_filter(signal_nose.signal, self.time_test)
                power_face, freqs_face = bandpass_filter(signal_face.signal, self.time_test)
                power_selection, freqs_selection, max_index = selection_signal(power_forehead, freqs_forehead,
                                                                               power_nose, freqs_nose, power_face,
                                                                               freqs_nose)
                self.bpms.append(60 * freqs_selection[max_index])
                print(60 * freqs_selection[max_index])
                # with open("data.txt", 'a') as file:
                # long += 1
                # file.write(str(long) + "\t" + str(60*freqs_selection[max_index]) + "\n")
                # axs[0].plot(freqs_forehead, power_forehead)
                # axs[1].plot(freqs_nose, power_nose)
                # axs[2].plot(freqs_face, power_face)

                # print(HR)
                # plt.figure()
                # plt.show()
                # plt.figure()
                # axs[2].plot(list(range(len(signal.signal))), signal.signal)
                # plt.show()

            # if len(raw_bvp_arr) == 100 :
            # app = QtWidgets.QApplication(sys.argv)
            # w = MainWindow(x = list(range(len(raw_bvp_arr)))  , y = raw_bvp_arr )
            # w.show()
            # sys.exit(app.exec_())

        self.frames_count += 1

    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frames_count = 0
        self.fps = 0
        self.mask = RoIExtraction()
        self.forehead_ROI = []
        self.nose_ROI = []
        self.face_ROI = []
        self.power_forehead = []
        self.freqs_forehead = []
        self.angle = []
        self.mean_previous = None
        self.raw_bvp_arr_forehead = []
        self.raw_bvp_arr_nose = []
        self.raw_bvp_arr_face = []
        self.time_begin_test = None
        self.time_end_test = None
        self.time_test = None
