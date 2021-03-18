import cv2
from display import Display
from extractor import Frame, denormalize, match_frames, IRt
import time
import numpy as np
import g2o
from multiprocessing import Process, Queue

import pangolin
import OpenGL.GL as gl

# Camera intrinsics
W, H = 1920//2, 1080//2
F = 270
K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))

display = Display(W, H)

# Global map
class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = Queue()
        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init()
        while True:
            self.viewer_refresh(q)

    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -30, -20, 0, 0, 0, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)
        
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)
        self.darr = None

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()
        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)


        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(ppts)

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(spts)

        pangolin.FinishFrame()

    
    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)

        for p in self.points:
            pts.append(p.pt)
        self.q.put((poses, pts))

mapp = Map()

class Point(object):
    # A Point is a 3-D point in the world
    # Each point is observed in multiple frames

    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []

        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        # Frame is the frame class
        self.frames.append(frame)
        self.idxs.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]


    # Reject points without enough "Parallax" and points behind the camera
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1,v1), 3, (0,255,0))
        cv2.line(img, (u1,v1), (u2, v2), (255,0,0))

    display.paint(img)

    # 3-D
    mapp.display()

if __name__== "__main__":
    cap = cv2.VideoCapture("test_countryroad.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break