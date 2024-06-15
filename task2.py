import cv2 as cv
import numpy as np
from collections import deque

center = None
frame_size = None
distance = 0

buffer = deque(maxlen=10)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_frame, (5, 5), 0)
    
    thresh = cv.threshold(blurred, 30, 2555, cv.THRESH_BINARY)[1]
    
    edged = cv.Canny(thresh, 30, 150)
    dilated = cv.dilate(edged, None, iterations=2)
    
    cnts, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv.contourArea, reverse=True)[0]
        M = cv.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)
        cv.drawContours(frame, [cnt], -1, (255, 0, 0), 3)
        cv.circle(frame, center, 7, (0, 255, 0), 2)
    
    if frame_size is None:
        height, width = gray_frame.shape[:2]
        frame_size = (width/2, height/2)   
        
    if center is not None:
        cv.putText(frame, f'({cx}, {cy})', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv.imshow('video', frame)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()