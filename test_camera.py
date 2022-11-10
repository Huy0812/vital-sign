from math import fabs
from pickle import TRUE
from pydoc import pager
from signal import signal
from statistics import mean
import sys
import cv2
import requests
from roi_detect import FaceMeshDetector
import numpy as np
import time
import base64
import datetime
import json
import random
import threading
import queue

from roi_selection import bandpass_filter
from signal_processing import Signal
url = 'http://127.0.0.1:3000/api/vitalsign/skin/v1.0'
video = cv2.VideoCapture(0)
fd = FaceMeshDetector(draw_landmarks=True)

buffer_length = 100
flag = False
session = random.randint(0,200)
rgb = 0
fps = 0
heartrate = []
signal = []
bvp_signal = []
image = None
count_frame = 0
terminate = False

def encode_frame(frame):
    return base64.b64encode(frame.astype(np.float32)).decode("utf-8")

def process_signal():
    global session,count,flag,rgb,fps,signal,terminate, heartrate, image
    count_frame = 0
    while (True):
        
        if flag == True:
            break

        time_start = time.perf_counter()


        ret,frame = video.read()

        if not ret:
            break
        
        # print(mean_grayscale(frame))
        # print(is_low_contrast(frame))

        ROI = fd(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "forehead")
        #print(ROI)
        ROI.draw_roi(frame)
        r,g,b = ROI.get_mean_rgb()
    

        total_time = (time.perf_counter() - time_start)
        if 1/25  - total_time > 0:
            time.sleep(abs(1/25 - total_time))
        # while (time.perf_counter() - time_start < 1/30-0.012):
        #     time.sleep(0.01)
        cv2.imshow('cam',frame)
        fps = int(1/(time.perf_counter() - time_start))
        if count_frame > 0:
            signal.append([r,g,b,fps])
            with open("test.txt", 'a') as file:
                file.write(str(g) + ",")
        count_frame += 1
        if image is None :
            image = frame 
        print('FPS:',fps)
        #time.sleep(0.1)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            terminate = True
            break

def send_request():

    global session,count,flag,rgb,fps,signal,terminate, heartrate, image
    count = 1
    while(True):
        time.sleep(0.01)
        
        if terminate == True:
                break
        # print(signal.qsize())
        if len(signal) % buffer_length == 0 and len(signal) != 0:
        
            start_rq = time.time()
            # buffer = json.dumps(signal)
            # base64EncodedStr = base64.b64encode(buffer.encode('utf-8'))
            # base64EncodedStr = base64EncodedStr.decode('utf-8')
            print('-----------------------------------------------------------')
            print('secsion:',session)
            if count == 9:
                flag = True
            print('Buffer so: ',count)
            print('do dai chuoi',len(signal))
            buffer = signal
            _, img_encoded = cv2.imencode('.jpg', image)
            img = img_encoded.tostring()
            #r = requests.post(url,data=json.dumps({"images": img,"device_id":"00-D2-79-51-B1-4A","buffer": buffer,"mode":"1","message_id":count,'finish':flag,'session_id':session}),headers={'Content-type': 'application/json', 'Accept': 'text/plain'}).json()
            #print('json: ',json.dumps({"device_id":"00-D2-79-51-B1-4A","buffer": signal,"mode":"1","message_id":count,'finish':flag,'session_id':session}))
            r = requests.post(url, data=img_encoded.tostring())
            print('thoi gian request:',time.time()-start_rq)
            #("Response Server is {}".format(r))
            print('-----------------------------------------------------------')
            #time_start = time.perf_counter()
            signal = []
            count += 1


            if count == 10:
                count = 0
                flag = False
                session = random.randint(6,100)
                break




# def vital_sign():
#     global session,count,flag,rgb,fps,signal,terminate, heartrate, bvp_signal
#     while (True):
#         time.sleep(1 / 1000000)
#         if len(bvp_signal) == 1000 :
#                 arr = []
#                 for i in range (0, len(heartrate) - 1) :
                    
#                     if abs(heartrate[i+1] - heartrate[i]) < 2 : 
#                         arr.clear()
#                         arr.append(heartrate[i])
#                         arr.append(heartrate[i+1])
                
#                 print("heart rate: ", mean(arr))
#                 sys.exit()
#         if len(bvp_signal) % 11 == 10 and len(bvp_signal) > 600 :
#             signal_forehead = Signal()
#             signal_forehead.signal = bvp_signal[-600:]
#             signal_forehead(600/fps)
#             power_forehead, freqs_forehead = bandpass_filter(signal_forehead.signal, 600/fps)
#             #print(power_forehead)
#             max_index_forehead = np.argmax(power_forehead)
#             heartrate.append(60 * freqs_forehead[max_index_forehead])
#             
#             #print("spo2", mean(self.spo2_face))
            
#         else:
#             pass

if __name__ == '__main__':
    t1 = threading.Thread(target=process_signal)
    t2 = threading.Thread(target=send_request)
    # t3 = threading.Thread(target=vital_sign)
    # # starting thread 1
    t1.start()
    t2.start()
    # t3.start()
   # time.sleep(1)
    # # wait until thread 1 is completely executed
    t1.join()
    t2.join()
    # t3.join()

    # both threads completely executed
    print("Done!")

