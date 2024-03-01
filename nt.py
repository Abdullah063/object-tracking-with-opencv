# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 00:12:50 2024

@author: ABDULLAH
"""

import cv2
import numpy as np

buffer_size = 16
blueLower = (70,  80,  0)
blueUpper = (170, 255, 255)

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    success, imgOriginal = cap.read()
    
    if success: 
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) 
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #cv2.imshow("hsv", hsv)
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255), 2)
            
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            cv2.circle(imgOriginal, center, 5, (255,0,255), -1)
            
  
        cv2.imshow("Orijinal Tespit", imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
