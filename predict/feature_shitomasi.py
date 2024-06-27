import cv2
from .parameters import Parameters  


class ShiTomasiDetector(object): 
    def __init__(self, num_features=Parameters.kNumFeatures, quality_level = 0.01, min_coner_distance = 3):
        self.num_features = num_features
        self.quality_level = quality_level
        self.min_coner_distance = min_coner_distance
        self.blockSize=5   

    def detect(self, frame, mask=None):                
        pts = cv2.goodFeaturesToTrack(frame, self.num_features, self.quality_level, self.min_coner_distance, blockSize=self.blockSize, mask=mask)
        if pts is not None: 
            kps = [ cv2.KeyPoint(p[0][0], p[0][1], self.blockSize) for p in pts ]
        else:
            kps = [] 
        return kps

