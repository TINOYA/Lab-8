import cv2 as cv
import numpy as np
from collections import deque

#инициализировать центр объекта и размеры рамки для расчета расстояния
center = None
frame_dimensions = None
distance = 0

#инициализировать буфер для хранения 10 измерений
buffer = deque(maxlen=10)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #преобразовать в оттенки серого и размыть изображение
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_frame, (5, 5), 0)
    
    #здесь подгон делать
    thresh = cv.threshold(blurred, 30, 255, cv.THRESH_BINARY)[1]
    
    #применить обнаружение и расширение тонких краев
    edged = cv.Canny(thresh, 30, 150)
    dilated = cv.dilate(edged, None, iterations=2)
    
    #найти контуры на изображении
    cnts, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        #отсортируйте контуры по площади и выберите самый большой
        cnt = sorted(cnts, key=cv.contourArea, reverse=True)[0]
        #вычислить центр объекта, используя моменты
        M = cv.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)
            
        #нарисуйте контур и центр на рамке
        cv.drawContours(frame, [cnt], -1, (255, 0, 0), 3)
        cv.circle(frame, center, 7, (0, 255, 0), 2)
    
    if frame_dimensions is None:
        #инициализировать размеры рамы для последующих расчетов
        height, width = gray_frame.shape[:2]
        frame_dimensions = (width/2, height/2)   #центр камеры
        
    if center is not None:
        distance = np.sqrt(((center[0] - frame_dimensions[0])**2 + (center[1] - frame_dimensions[1])**2))
        buffer.append(distance)
        #рассчитать среднее расстояние от объекта до центра видео
        avg_distance = np.mean(buffer) if len(buffer) == 10 else None
    
    #отобразить рамку с контурами и расстоянием
    cv.putText(frame, f'Distance: {avg_distance}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv.imshow('video', frame)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()