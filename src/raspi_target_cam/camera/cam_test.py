import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set()
w,h = 3264, 2448
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)



i = 0
while cap.isOpened():
    ret, frame = cap.read(cv.CAP_DSHOW)

    i += 1

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
