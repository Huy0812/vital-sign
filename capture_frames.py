from re import M
import cv2
import numpy as np
import torch
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from raw_bvp import raw_bvp
from roi_extraction import RoIExtraction
from luminance import mean_grayscale, is_low_contrast, is_Y_channel
from roi_selection import bandpass_filter, selection_signal
from signal_processing import Signal


class CaptureFrames():

    def __init__(self):
        self.frame_count = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = RoIExtraction()

    def __call__(self, source):
        # self.pipe = pipe
        self.capture_frames(source)

    def capture_frames(self, source):
        camera = cv2.VideoCapture(source)
        time.sleep(1)
        (grabbed, frame) = camera.read()
        time_begin = time.time()
        self.frames_count = 0
        mean_previous = None
        raw_bvp_arr_forehead = []
        raw_bvp_arr_nose = []
        raw_bvp_arr_face = []

        time_begin_test = None
        time_end_test = None
        time_test = None
        long = 0
        while grabbed:

            (grabbed, frame) = camera.read()

            if not grabbed:
                continue

            k = cv2.waitKey(1)

            if k != -1:
                self.terminate(camera)
                break

            (check_y_channel, mean_y_channel) = is_Y_channel(frame, mean_previous)
            # frame not satisfy luminance condition
            if (is_low_contrast(frame) or not check_y_channel):
                time_begin_test = None
                raw_bvp_arr_face.clear()
                raw_bvp_arr_forehead.clear()
                raw_bvp_arr_nose.clear()
            else:
                (check_luminance, orig) = mean_grayscale(frame)
                if (not (check_luminance)):
                    continue
                else:
                    #time_begin_mask = time.time()
                    self.mask(frame)
                    #time_end_mask = time.time()
                    #print(time_begin_mask-time_end_mask)
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
                    raw_bvp_arr_forehead.append(forehead_BVP)
                    raw_bvp_arr_nose.append(nose_BVP)
                    raw_bvp_arr_face.append(face_BVP)
                    if (time_begin_test == None and len(raw_bvp_arr_forehead) >= 100):
                        time_begin_test = time.time()
                    time_end_test = time.time()

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

            mean_previous = mean_y_channel
            if self.frames_count % 30 == 29:
                time_end = time.time()
                sys.stdout.write(f'\rFPS: {30 / (time_end - time_begin)}')
                sys.stdout.flush()
                time_begin = time.time()
            if (time_begin_test != None and time_end_test != None):
                if len(raw_bvp_arr_forehead) >= 300:
                    if (time_test == None):
                        time_test = time_end_test - time_begin_test
                    # if len(raw_bvp_arr_face) == 600 :
                    signal_forehead = Signal()
                    signal_nose = Signal()
                    signal_face = Signal()
                    signal_forehead.signal = raw_bvp_arr_forehead[-200:]
                    signal_nose.signal = raw_bvp_arr_nose[-200:]
                    signal_face.signal = raw_bvp_arr_face[-200:]
                    fig, axs = plt.subplots(5)
                    axs[0].plot(range(len(signal_forehead.signal)), signal_forehead.signal)

                    signal_forehead(time_test)
                    signal_nose(time_test)
                    signal_face(time_test)
                    axs[1].plot(range(len(signal_forehead.signal)), signal_forehead.signal)


                    power_forehead, freqs_forehead = bandpass_filter(signal_forehead.signal, time_test)
                    power_nose, freqs_nose = bandpass_filter(signal_nose.signal, time_test)
                    power_face, freqs_face = bandpass_filter(signal_face.signal, time_test)
                    power_selection, bpm_selection, max_index = selection_signal(power_forehead, freqs_forehead,
                                                                                   power_nose, freqs_nose, power_face,
                                                                                   freqs_face)

                    print(60*bpm_selection[max_index])
                    with open("data.txt", 'a') as file:
                        long += 1
                        file.write(str(long) + "\t" + str(60*bpm_selection[max_index]) + "\n")
                    axs[2].plot(freqs_forehead, power_forehead)
                    axs[3].plot(freqs_nose, power_nose)
                    axs[4].plot(freqs_face, power_face)

                    # print(HR)
                    # plt.figure()
                    # plt.show()
                    # plt.figure()
                    # axs[2].plot(list(range(len(signal.signal))), signal.signal)
                    plt.show()
            
            # if len(raw_bvp_arr) == 100 :
            # app = QtWidgets.QApplication(sys.argv)
            # w = MainWindow(x = list(range(len(raw_bvp_arr)))  , y = raw_bvp_arr )
            # w.show()
            # sys.exit(app.exec_())

            self.frames_count += 1

        self.terminate(camera)

    def terminate(self, camera):
        # self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()