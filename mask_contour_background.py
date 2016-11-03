import cv2
import numpy as np

try:
    vcap = cv2.VideoCapture("rtsp://192.168.1.15:8554/")
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50*40, detectShadows=True)
    while(vcap.isOpened()):
        ret, frame = vcap.read()
        if not ret: break
        fgmask = fgbg.apply(frame)
        masked = cv2.bitwise_and(frame, frame, mask=fgmask)
        ret, thresh = cv2.threshold(masked, 64, 64, 64)
        threshgrey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        _, contours, hier = cv2.findContours(threshgrey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hulls = [cv2.convexHull(contour) for contour in contours]
        contour_mask = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(contour_mask, hulls, -1, (255, 255, 255), -1)
        blurred = cv2.blur(contour_mask, (10,10))
        im = cv2.bitwise_and(frame, blurred)
        cv2.imshow('im', im)
        cv2.imshow('frame', frame)
        cv2.imshow('masked', masked)
        cv2.imshow('contours', blurred)
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
finally:
    vcap.release()
    cv2.destroyAllWindows()

