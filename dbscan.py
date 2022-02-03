import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN  # For DBSCAN
import numpy as np

def get_files(path):

    files = os.listdir(path)
    files = [os.path.join(path, file_) for file_ in files]
    files.sort()
    print('files len:',len(files))
    return files

class DbscanTamper:
    """ Class for getting orb keypoints and clustering using DBSCAN to detect forgery"""
    def __init__(self, showflag=True):
        self.orb = cv2.ORB_create()
        self.showflag = showflag

    def kpDetector(self, img):
        kp = self.orb.detect(img, None)
        kp, des = self.orb.compute(img, kp)
        return kp, des

    def show(self, img, win='img'):
        cv2.imshow(win,img)
        cv2.waitKey(0)

    def show_kp(self, img, kp):
        img2 =  cv2.drawKeypoints(img, kp, None, color=(255,0,0))
        cv2.imshow('kp',img2)
        cv2.waitKey(0)

    def make_clusters(self, de,eps=40,min_sample=2):
        clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(de)
        return clustering

    def locate_forgery(self, img,clustering,kps):
        forgery=img.copy()
        clusters = [[] for i in range(np.unique(clustering.labels_).shape[0]-1)]
        for idx in range(len(kps)):
            if clustering.labels_[idx]!=-1:
                clusters[clustering.labels_[idx]].append((int(kps[idx].pt[0]),int(kps[idx].pt[1])))
        forg_flag = False
        for points in clusters:
            if len(points)>1:
                for idx1 in range(len(points)):
                    for idx2 in range(idx1+1,len(points)):
                        if self.showflag:
                            cv2.line(forgery,points[idx2],points[idx1],(255,0,0),5)
                        forg_flag = True
        if self.showflag:
            self.show(forgery, 'forgery:')
            cv2.destroyAllWindows()

        return forg_flag


    def det_forgery(self, img):
        key_points,descriptors=self.kpDetector(img)
        if self.showflag:
            self.show_kp(img, key_points)

        clusters=self.make_clusters(descriptors)
        forg_flag = self.locate_forgery(img,clusters,key_points)
        return forg_flag

if __name__=='__main__':  
    sbscan_obj = DbscanTamper(showflag = True)
    org_path = 'data/org'
    tamp_path = 'data/tamp'

    org_files = get_files(org_path)
    tamp_files = get_files(tamp_path)

    print('org_files',org_files)
    print('tamp_files',tamp_files)

    if 1:
        # Img 1
        file_  = tamp_files[1]
        tampered = cv2.imread(file_)
        print('file:', file_)
        forg_flag = sbscan_obj.det_forgery(tampered)
        print('Forgery:', forg_flag)

    if 1:
        # Img 2
        file_  = org_files[1]
        org = cv2.imread(file_)
        print('file:', file_)
        forg_flag = sbscan_obj.det_forgery(org)
        print('Forgery:', forg_flag)