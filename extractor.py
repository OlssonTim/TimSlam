import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractRt(E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]])
    U,d,Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    Rt = np.concatenate([R,t.reshape(3,1)], axis=1)
    return Rt

class Extractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.dot(self.K, [pt[0], pt[1], 1.0])
        return int(round(ret[0])), int(round(ret[1]))

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
        Rt = None
        if len(ret) > 0:
            ret = np.array(ret)
            # Normalize coords: Subtract to move to 0
            print(img.shape)
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])
            # ret[:, 1, :] = np.dot(self.Kinv, add_ones(ret[:,1,:]).T).T[:, 0:2]

            model, inliers = ransac((ret[:, 0], 
                                    ret[:, 1]), EssentialMatrixTransform, min_samples=8, residual_threshold=0.005, max_trials=200)
            
            ret = ret[inliers]

            Rt = extractRt(model.params)

        self.last = {'kps': kps, 'des': des}

        return ret, Rt

