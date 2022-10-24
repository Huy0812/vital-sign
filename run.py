from multithread import CaptureFrames

if __name__=="__main__":
    file = open("data.txt","r+")
    file.truncate(0)
    file.close()
    capture = CaptureFrames()
    capture()