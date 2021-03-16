import cv2
from display import Display
from extractor import Extractor
import time
import numpy as np

W = 1920//2
H = 1080//2

display = Display(W, H)

fe = Extractor()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches = fe.extract(img)

    if matches is None:
        return

    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1,v1), 3, (0,255,0))
        cv2.line(img, (u1,v1), (u2, v2), (255,0,0))

    display.paint(img)

if __name__== "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break