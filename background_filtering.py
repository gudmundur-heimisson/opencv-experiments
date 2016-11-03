import cv2
try:
    vcap = cv2.VideoCapture("rtsp://192.168.1.15:8554/")
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=81)
    while(vcap.isOpened()):
        ret, frame = vcap.read()
        if not ret: break
        fgmask = fgbg.apply(frame)
        masked = cv2.bitwise_and(frame, frame, mask=fgmask)
        cv2.imshow('masked', masked)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
finally:
    vcap.release()
    cv2.destroyAllWindows()

