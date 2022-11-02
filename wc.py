import cv2

cap = cv2.VideoCapture(1)

while True:
    r, frame = cap.read()
    if not r: break 
    
    cv2.imshow('img', frame)
    if cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()