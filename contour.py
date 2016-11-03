import cv2
import numpy as np

try:
    vcap = cv2.VideoCapture("rtsp://192.168.1.15:8554/")
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(vcap.isOpened()):
        ret, frame = vcap.read()
        if not ret: break
        ret, thresh = cv2.threshold(frame, 127, 255, 0)
        threshgrey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        _, contours, hier = cv2.findContours(threshgrey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(cnt, contours, -1, (255, 255, 255), -1)
        masked = cv2.bitwise_and(frame, cnt)
        blurred = cv2.blur(masked, (5,5))
        cv2.imshow('frame', frame)
        cv2.imshow('cnt', cnt)
        cv2.imshow('masked', masked)
        cv2.imshow('thresh', threshgrey)
        cv2.imshow('blurred', blurred)
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
finally:
    vcap.release()
    cv2.destroyAllWindows()

