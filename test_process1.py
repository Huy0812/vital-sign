from concurrent.futures import process, thread
from email import message
import json
import sys
import argparse
from unittest import result
import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from roi_detect import FaceMeshDetector
#from rppg.camera import Camera
from scipy import signal
import heartpy as hp
import matplotlib.pyplot as plt
from csv import DictWriter
import cv2
#from rppg.filters import get_butterworth_filter
from scipy.signal import butter, lfilter
#from rppg.processors.chrom import ChromProcessor
#from rppg.processors.color_mean import ColorMeanProcessor
#from rppg.processors.processor import FilteredProcessor
#from rppg.rppg import RPPG
#from rppg.hr import from_peaks,from_fft, from_red_and_blue
# from rppg.hr import HRCalculator
#from rppg.camera import Camera
import time
import datetime
from threading import Thread
import math
import sys
#from rppg.hr import HRCalculator
from process import CaptureFrames

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

app = Flask(__name__)
# creating an API object
api = Api(app)

POST_HEADERS = {'content-type': 'application/json'}

Infor_request = {}
        
#digital_lowpass = get_butterworth_filter(25, 1.5)
cutoff = list(map(float,"0.7,4".split(",")))
#digital_bandpass = get_butterworth_filter(25,cutoff=cutoff, btype="bandpass")
#hr_calc = HRCalculator(parent=None, update_interval=25, winsize=300,filt_fun=lambda vs: [digital_lowpass(v) for v in vs])
#processor = ChromProcessor(winsize=10,method="xovery")
#processor = FilteredProcessor(processor,digital_bandpass)
  
  
  
# another resource to calculate the square of a number
class Facial_Skin (Resource) :
    def __init__(self):
  
        self.hr = []
        self.spo2 = []
        self.rr = []
        self.dark_circle = []
        self.device_id = None
        self.mean_rgb_ts = None
        self.ts = None
        self.vs = None
        self.winsize = 100
        self.vs_red = None
        self.vs_blue =  None

    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def get(self):
        return jsonify({'out':'ok'})

    def post(self) :
        data = request.get_json(force=True)
        # decode base64
        self.mean_rgb_ts = data['buffer']
        #self.mean_rgb_ts = decode_buffer(self.mean_rgb_ts).tolist()
        # print(self.mean_rgb_ts)

        # infor request
        self.device_id = data['device_id']
        print(self.device_id)
        mode_process = data['mode']
        self.message_id = data['message_id']
        status_device = data['finish']
        self.session_id = data['session_id']
        nparr = np.fromstring(r.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("hello")
        time.sleep(0.001)
        self.process = CaptureFrames()
        fd = FaceMeshDetector(draw_landmarks=True)
        rule_roi = fd(image, "RULE")
        rarule_roi = fd(image, "RARULE")
        rure_roi = fd(image, "RURE")
        rarure_roi = fd(image,"RARURE")
            #receive frame
        self.process.dark_circle(image, rule_roi, rarule_roi, rure_roi, rarure_roi)
        self.dark_circle = self.process.darkCircle
        print('********************', self.dark_circle)
        Infor_request[self.device_id][self.session_id]['results']= {'dark_circle': self.dark_circle}
        self.reset()
        return jsonify(Infor_request[self.device_id][self.session_id]['results'])
    def reset(self):
        Infor_request[self.device_id][self.session_id]['buffer_vs'] = []
        Infor_request[self.device_id][self.session_id]['buffer_ts'] = []
        Infor_request[self.device_id][self.session_id]['buffer_blue'] = []
        Infor_request[self.device_id][self.session_id]['buffer_red'] = []
class Facial_HRate(Resource):
    
    def __init__(self):
  
        self.hr = []
        self.spo2 = []
        self.rr = []
        self.dark_circle = []
        self.device_id = None
        self.mean_rgb_ts = None
        self.ts = None
        self.vs = None
        self.winsize = 100
        self.vs_red = None
        self.vs_blue =  None

    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def get(self):
        
        return jsonify({'out':'ok'})
  
    def post(self):
        data = request.get_json(force=True)
        # decode base64
        self.mean_rgb_ts = data['buffer']
        #self.mean_rgb_ts = decode_buffer(self.mean_rgb_ts).tolist()
        # print(self.mean_rgb_ts)

        # infor request
        self.device_id = data['device_id']
        print(self.device_id)
        mode_process = data['mode']
        self.message_id = data['message_id']
        status_device = data['finish']
        self.session_id = data['session_id']

     
        if self.device_id not in Infor_request.keys():
            Infor_request[self.device_id]={}
        if self.session_id not in Infor_request[self.device_id].keys():
            Infor_request[self.device_id][self.session_id] = {'all_buffer':[],'results':0}
            
        if self.message_id not in Infor_request[self.device_id][self.session_id].keys():
            Infor_request[self.device_id][self.session_id][self.message_id] = {'msg_buffer':[],'mode':0}

        Infor_request[self.device_id][self.session_id][self.message_id]['mode'] = mode_process
        Infor_request[self.device_id][self.session_id][self.message_id]['msg_buffer'] = self.mean_rgb_ts 

        Thread(target = self.create_buffer).start()
        if(status_device == False):
            return jsonify({'out':self.mean_rgb_ts})
           

            
        else:
            time.sleep(0.001)
            all_buffer = Infor_request[self.device_id][self.session_id]['all_buffer']
            print('do dai buffer: ',len(all_buffer))


            cam = Camera(buffer_data= all_buffer)
            self.rppg = RPPG(camera=cam,msg_id=self.message_id,hr_calculator=hr_calc)
            self.rppg.add_processor(processor)
            for c in "rgb":
                self.rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1)) 
            self.rppg.start()
            ts = self.rppg.get_ts(900)
            vs = next(self.rppg.get_vs(900))

            indexes =[]
            for i in range (len(vs[100:])):
                indexes.append(i)
            plt.plot(indexes,vs[100:],'r')
            plt.axvline(x =100 , color = 'b', label = 'axvline - full height')
            plt.axvline(x =200 , color = 'b', label = 'axvline - full height')
            plt.axvline(x =300 , color = 'b', label = 'axvline - full height')
            plt.axvline(x =400 , color = 'b', label = 'axvline - full height')
            plt.axvline(x =500 , color = 'b', label = 'axvline - full height')
            plt.axvline(x =600 , color = 'b', label = 'axvline - full height')
            plt.axvline(x =700 , color = 'b', label = 'axvline - full height')
            plt.axvline(x =800 , color = 'b', label = 'axvline - full height')

            plt.savefig(f'./ngocpt/{self.device_id}_facial_signal.png')
            plt.close()

            self.hr = self.rppg.hr_calculator.bpms
            self.rr = self.rppg.hr_calculator.rrs
            self.spo2 = self.rppg.hr_calculator.spo2s
            print(self.hr)
            # print('---------------------------------------')
            hr_clean = self.outliner(self.hr[-10:])
            print(hr_clean)
            print(np.mean(hr_clean))

            #self.rppg.finish()

            Infor_request[self.device_id][self.session_id]['results']= {'hr': np.mean(hr_clean), 'rr': np.mean(self.rr), 'spo2': np.mean(self.spo2)}
            self.reset()

            return jsonify(Infor_request[self.device_id][self.session_id]['results'])


    def outliner(self,arr,k=2):

        elements = np.array(arr) 
        mean = np.mean(elements, axis=0)
        sd = np.std(elements, axis=0)
        final_list = [x for x in arr if (x > mean - k * sd)]
        print('min:',mean - k * sd)
        final_list = [x for x in final_list if (x < mean + k * sd)]
        print('max:',mean + k * sd)
        return  final_list

    def reset(self):
        Infor_request[self.device_id][self.session_id]['buffer_vs'] = []
        Infor_request[self.device_id][self.session_id]['buffer_ts'] = []
        Infor_request[self.device_id][self.session_id]['buffer_blue'] = []
        Infor_request[self.device_id][self.session_id]['buffer_red'] = []

    def create_buffer(self):

        # if self.message_id > 1: 
           
        #print(self.mean_rgb_ts)
        start_time = time.time()
        # cam = Camera(buffer_data= Infor_request[self.device_id][self.session_id][self.message_id]['msg_buffer'])
        # self.rppg = RPPG(camera=cam,msg_id=self.message_id)
        # self.rppg.add_processor(processor)
        # for c in "rgb":
        #     self.rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1)) 
            
        # self.rppg.start()
        # self.ts = self.rppg.get_ts(self.winsize)
        # self.vs = next(self.rppg.get_vs(self.winsize))

        # print(len(self.ts),len(self.vs))

        # self.vs_red = self.rppg._processors[1].vs[-self.winsize:]
        # self.vs_blue = self.rppg._processors[3].vs[-self.winsize:]
        
        Infor_request[self.device_id][self.session_id]['all_buffer'] = Infor_request[self.device_id][self.session_id]['all_buffer'] + self.mean_rgb_ts
        
        # print('Estimation time: ',time.time()-start_time)
        # print('-----------------------------------------------')

class Fingertip_HRate(Resource):
    
    def __init__(self):
  
        self.hr = None
        self.spo2 = None
        self.RR = None
        self.ts = None
        self.vs_red = None

        self.message_id = None
        self.device_id = None
        
    def get(self):
        
        return jsonify({'message': 'VitalSign'})
    
    def butter_bandpass(self, lowcut, highcut, fs, order=2):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def preprocess_signal(self, hb_list, ts):
        hb_list = self.butter_bandpass_filter(hb_list, 0.5, 4, 30, order=1)
        # hb_list = hp.enhance_peaks(hb_list, iterations=3)
        # even_times = np.linspace(ts[0], ts[-1], len(hb_list))
        # processed = signal.detrend(hb_list)#detrend the signal to avoid interference of light change
        plt.plot(ts, hb_list)
        plt.savefig("finger_fig.png")
        plt.close()

        # interpolated = np.interp(even_times, ts, processed) #interpolation by 1
        # interpolated = np.hamming(len(hb_list)) * interpolated#make the signal become more periodic (advoid spectral leakage)
        # norm = interpolated/np.linalg.norm(interpolated)
        norm = np.array(hb_list)
        ts = np.array(ts)
        
        return norm
    
    def post(self):
        data = request.get_json(force=True)

        # decode base64
        self.vs_red = data['buffer']
        #self.vs_red = decode_buffer(self.vs_red).tolist()

        # infor request
        self.device_id = data['device_id']
        mode_process = data['mode']
        self.message_id = data['message_id']
        status_device = data['finish']
        self.session_id = data['session_id']

     
        if self.device_id not in Infor_request.keys():
            Infor_request[self.device_id]={}
        if self.session_id not in Infor_request[self.device_id].keys():
            Infor_request[self.device_id][self.session_id] = {'buffer_vs':[],'buffer_ts':[],'results':0}
            
        if self.message_id not in Infor_request[self.device_id][self.session_id].keys():
            Infor_request[self.device_id][self.session_id][self.message_id] = {'msg_buffer':[],'mode':0}

        Infor_request[self.device_id][self.session_id][self.message_id]['mode'] = mode_process
        Infor_request[self.device_id][self.session_id][self.message_id]['msg_vs'] = self.vs_red 
        Infor_request[self.device_id][self.session_id]['buffer_vs'] += self.vs_red
        #print(self.vs_red)
        # Thread(target = self.create_buffer).start()
        
        if(status_device == False):
            return 'Request Done'
            
        else:
            
            ts = []
            red_list = []
            field_names = ['Signal', 'Timestamp']
            start_time = time.time()
            for ind, bf in enumerate(Infor_request[self.device_id][self.session_id]['buffer_vs']):
                # print(bf)
                ts.append(ind/30)
                red_list.append(bf[0])
                
            red_list = np.array(255 - np.array(red_list))
            ID_ts = np.array(ts)
            
            red_list = red_list[150:]
            ID_ts = ID_ts[150:]
            
            dict = {'Signal': red_list, 'Timestamp': ID_ts}
            
            with open('event.csv', 'a') as f_object:
                # Pass the file object and a list
                # of column names to DictWriter()
                # You will get a object of DictWriter
                dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            
                # Pass the dictionary as an argument to the Writerow()
                dictwriter_object.writerow(dict)
            
                # Close the file object
                f_object.close()

            sample_rate = len(ID_ts) / (ID_ts[-1] - ID_ts[0])
            norm_red = self.preprocess_signal(red_list, ID_ts)
            
            wd, m = hp.process(norm_red, sample_rate = sample_rate, windowsize=0.75)
            #plot_object = hp.plotter(wd, m, show=False)

            # plot_object.savefig('hp_plot.jpg') #saves the plot as JPEG image.
            # plot_object.close()

            print("------------------------------------")
            print("Process time:", time.time() - start_time)
            print("------------------------------------")
            # SpO2 = self.calculate_SpO2(red_list, blue_list)
            # m['SpO2'] = SpO2
            
            self.hr = m['bpm']
            self.spo2 = 99
            # self.RR = None
            self.RR = m['breathingrate'] * 60
            # ok r đấy 
            # save results
            Infor_request[self.device_id][self.session_id]['results']= {'hr':self.hr, 'rr':self.RR, 'spo2':self.spo2}
            return jsonify(Infor_request[self.device_id][self.session_id]['results'])



if __name__ == '__main__':
    # adding the defined resources along with their corresponding urls
    api.add_resource(Facial_HRate, '/api/vitalsign/face/v1.0')
    api.add_resource(Fingertip_HRate, '/api/vitalsign/finger/v1.0')
    api.add_resource(Facial_Skin, '/api/vitalysign/skin/v1.0')
    app.run(host='0.0.0.0',port=1999, debug = True, threaded=True)