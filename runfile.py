from capture_frames import CaptureFrames
from optparse import OptionParser
import time

class RunPOS():
    def __init__(self, sz=270, fs=28, bs=30):
        self.batch_size = bs
        self.frame_rate = fs
        self.signal_size = sz

    def __call__(self, source):
        time1 = time.time()
        capture = CaptureFrames()
        capture(source)
        time2 = time.time()
        print(f' time {time2 - time1}')

def get_args():
    parser = OptionParser()
    parser.add_option('-s', '--source', dest='source', default="D:/Python/angelo/angelo_resting/cv_camera_sensor_stream_handler.avi",
                        help='Signal Source: 0 for webcam or file path')

    (options, _) = parser.parse_args()
    return options

if __name__=="__main__":
    args = get_args()
    source = args.source
    file = open("data.txt","r+")
    file.truncate(0)
    file.close()
    runPOS = RunPOS()
    runPOS(source)
    # capture = CaptureFrames()
    # capture(0)