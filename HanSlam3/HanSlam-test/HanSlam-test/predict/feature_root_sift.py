from enum import Enum
import numpy as np 

class RootSIFTFeature2D:
    def __init__(self, feature):
        self.feature = feature

    def detect(self, frame, mask=None):
        return self.feature.detect(frame, mask)
 
    def transform_descriptors(self, des, eps=1e-7): 
        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)        
        return des 
            
    def compute(self, frame, kps, eps=1e-7):
        (kps, des) = self.feature.compute(frame, kps)

        if len(kps) == 0:
            return ([], None)
        des = self.transform_descriptors(des)

        return (kps, des)

    def detectAndCompute(self, frame, mask=None):
        (kps, des) = self.feature.detectAndCompute(frame, mask)

        if len(kps) == 0:
            return ([], None)

        des = self.transform_descriptors(des)

        return (kps, des)
