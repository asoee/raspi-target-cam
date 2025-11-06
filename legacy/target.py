import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity

cap = cv.VideoCapture("C:\\Users\\ander\\Downloads\\pistol-s-1.mp4")

lastframe = []
ltres = None

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    i += 1

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    x, y = 180, 100
    w, h = 900, 900
    crop_img = frame[y:y + h, x:x + w]

    pts1 = np.float32([[25, 25], [900, 0], [0, 900], [875, 875]])
    pts2 = np.float32([[0, 0], [900, 0], [0, 900], [900, 900]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(crop_img, M, (900, 900))

    # extract red channel
    red_channel = dst.copy()
    red_channel[:, :, 0] = 0  # Set the Blue channel to 0
    red_channel[:, :, 1] = 0  # Set the Green channel to 0
    green_channel = dst.copy()
    red_channel[:, :, 0] = 0  # Set the Blue channel to 0
    red_channel[:, :, 2] = 0  # Set the Red channel to 0
    blue_channel = dst.copy()
    red_channel[:, :, 2] = 0  # Set the Red channel to 0
    red_channel[:, :, 1] = 0  # Set the Green channel to 0
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    # gray = red_channel

    # gray = cv.GaussianBlur(gray, (5, 5), 0)
    gray = cv.medianBlur(gray, 5)
    ctres = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    btres = cv.threshold(gray, 38, 255, cv.THRESH_BINARY)[1]
    if len(lastframe) > 0:
        if i % 1 == 0:
            (score, diff) = structural_similarity(lastframe, gray, full=True)
            if score < 0.97:
                print("Image Similarity at {}: {:.4f}%".format(i, score * 100))

                # The diff image contains the actual image differences between the two images
                # and is represented as a floating point data type in the range [0,1]
                # so we must convert the array to 8-bit unsigned integers in the range
                # [0,255] before we can use it with OpenCV
                diff = (diff * 255).astype("uint8")
                diff_box = cv.merge([diff, diff, diff])

                # Threshold the difference image, followed by finding contours to
                # obtain the regions of the two input images that differ
                # thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
                # contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # contours = contours[0] if len(contours) == 2 else contours[1]

                # cv.imshow('diff', diff)
                # cv.imshow('thres', thresh)
                # cv.imshow('last', lastframe)
                cv.imshow('current', gray)
                # cv.imshow('ctres', ctres)
                cv.imshow('red_channel', dst[:, :, 2])
                cv.imshow('green_channel', dst[:, :, 1])
                cv.imshow('blue_channel', dst[:, :, 0])
                # cv.imshow('ltres', ltres)
                cv.imshow('btres', btres)
                # if cv.waitKey(0) == ord('q'):
                #     break
                # break

    lastframe = gray
    ltres = ctres
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
