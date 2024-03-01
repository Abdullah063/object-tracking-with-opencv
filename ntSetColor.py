import cv2
import numpy as np

buffer_size = 16
blueLower = (100,  100,  0)
blueUpper = (170, 255, 255)

# Boş bir callback fonksiyonu
def nothing(x):
    pass

# Pencere oluştur
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Hue Lower', 'Trackbars', blueLower[0], 180, nothing)
cv2.createTrackbar('Hue Upper', 'Trackbars', blueUpper[0], 180, nothing)
cv2.createTrackbar('Saturation Lower', 'Trackbars', blueLower[1], 255, nothing)
cv2.createTrackbar('Saturation Upper', 'Trackbars', blueUpper[1], 255, nothing)
cv2.createTrackbar('Value Lower', 'Trackbars', blueLower[2], 255, nothing)
cv2.createTrackbar('Value Upper', 'Trackbars', blueUpper[2], 255, nothing)

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    success, imgOriginal = cap.read()
    
    if success: 
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) 
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Trackbar değerlerini al
        h_lower = cv2.getTrackbarPos('Hue Lower', 'Trackbars')
        h_upper = cv2.getTrackbarPos('Hue Upper', 'Trackbars')
        s_lower = cv2.getTrackbarPos('Saturation Lower', 'Trackbars')
        s_upper = cv2.getTrackbarPos('Saturation Upper', 'Trackbars')
        v_lower = cv2.getTrackbarPos('Value Lower', 'Trackbars')
        v_upper = cv2.getTrackbarPos('Value Upper', 'Trackbars')
        
        blueLower = (h_lower, s_lower, v_lower)
        blueUpper = (h_upper, s_upper, v_upper)
        
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
            
        cv2.imshow("Original Detection", imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
