import cv2 
camera = cv2.VideoCapture(0)
while True :
    grabed, frame = camera.read() 
    cv2.imshow('img', frame)
    cv2.waitKey(1)