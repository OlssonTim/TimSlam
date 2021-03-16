import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class Extractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):
        # Detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=-1).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        # Extraction
        kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # Matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # Filter
        if len(ret) > 0:
            ret = np.array(ret)

            model, inliers = ransac((ret[:, 0], 
                                    ret[:, 1]), FundamentalMatrixTransform, min_samples=8, residual_threshold=1, max_trials=100)

            ret = ret[inliers]

        self.last = {'kps': kps, 'des': des}

        return ret

