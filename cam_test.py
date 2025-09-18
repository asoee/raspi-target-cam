import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity

cap = cv.VideoCapture(1,cv.CAP_DSHOW)
cap.set()


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
