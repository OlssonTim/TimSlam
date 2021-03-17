import cv2
from display import Display
from extractor import Frame, denormalize, match
import time
import numpy as np
import g2o

# Camera intrinsics
W, H = 1920//2, 1080//2
F = 270
K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))

display = Display(W, H)

frames = []
def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(img, K)
    frames.append(frame)

    if len(frames) <= 1:
        return

    ret, Rt = match(frames[-2], frames[-1])

    for pt1, pt2 in ret:
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

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