import numpy as np 
import cv2
from parameters import Parameters  
from enum import Enum
from collections import defaultdict


kRatioTest = Parameters.kFeatureMatchRatioTest
kVerbose = False 

class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1     
    FLANN = 2

def feature_matcher_factory(norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest, type=FeatureMatcherTypes.FLANN):
    if type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type == FeatureMatcherTypes.FLANN:
        return FlannFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    return None 

class FeatureMatcher(object): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.BF):
        self.type = type 
        self.norm_type = norm_type 
        self.cross_check = cross_check   # apply cross check 
        self.matches = []
        self.ratio_test = ratio_test 
        self.matcher = None 
        self.matcher_name = ''
        
        

    def match(self, des1, des2, ratio_test=None):
        if kVerbose:
            print(self.matcher_name,', norm ', self.norm_type)      
        matches = self.matcher.knnMatch(des1, des2, k=2)  
        self.matches = matches
        return self.goodMatches(matches, des1, des2, ratio_test)          
    
    def matchWithCrossCheckAndModelFit(self, des1, des2, kps1, kps2, ratio_test=None, cross_check=True, err_thld=1, info=''):
        idx1, idx2 = [], []          
        if ratio_test is None: 
            ratio_test = self.ratio_test
            
        init_matches1 = self.matcher.knnMatch(des1, des2, k=2)
        init_matches2 = self.matcher.knnMatch(des2, des1, k=2)

        good_matches = []

        for i,(m1,n1) in enumerate(init_matches1):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[m1.trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio_test is not None:
                cond2 = m1.distance <= ratio_test * n1.distance
                cond *= cond2
            if cond:
                good_matches.append(m1)
                idx1.append(m1.queryIdx)
                idx2.append(m1.trainIdx)

        if type(kps1) is list and type(kps2) is list:
            good_kps1 = np.array([kps1[m.queryIdx].pt for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx].pt for m in good_matches])
        elif type(kps1) is np.ndarray and type(kps2) is np.ndarray:
            good_kps1 = np.array([kps1[m.queryIdx] for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx] for m in good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        ransac_method = None 
        try: 
            ransac_method = cv2.USAC_MSAC 
        except: 
            ransac_method = cv2.RANSAC
        _, mask = cv2.findFundamentalMat(good_kps1, good_kps2, ransac_method, err_thld, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
        print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        return idx1, idx2, good_matches, mask
     
    def goodMatchesOneToOne(self, matches, des1, des2, ratio_test=None):
        idx1, idx2 = [], []  
        if ratio_test is None: 
            ratio_test = self.ratio_test
        if matches is not None:         
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)   
            index_match = dict()  
            for m, n in matches:
                if m.distance > ratio_test * n.distance:
                    continue     
                dist = dist_match[m.trainIdx]
                if dist == float_inf: 
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2)-1
                else:
                    if m.distance < dist: 
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx) 
                        idx1[index]=m.queryIdx
                        idx2[index]=m.trainIdx                        
        return idx1, idx2

    def goodMatchesSimple(self, matches, des1, des2, ratio_test=None):
        idx1, idx2 = [], []   
        if ratio_test is None: 
            ratio_test = self.ratio_test            
        if matches is not None: 
            for m,n in matches:
                if m.distance < ratio_test * n.distance:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)                                                         
        return idx1, idx2 

    def goodMatches(self, matches, des1, des2, ratio_test=None): 
        return self.goodMatchesOneToOne(matches, des1, des2, ratio_test)


class BfFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.BF):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        self.matcher = cv2.BFMatcher(norm_type, cross_check)     
        self.matcher_name = 'BfFeatureMatcher'   

class FlannFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.FLANN):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        if norm_type == cv2.NORM_HAMMING:
            FLANN_INDEX_LSH = 6
            self.index_params= dict(algorithm = FLANN_INDEX_LSH,  
                        table_number = 6,      
                        key_size = 12,         
                        multi_probe_level = 1)           
        if norm_type == cv2.NORM_L2: 
            FLANN_INDEX_KDTREE = 1
            self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)  
        self.search_params = dict(checks=32)          
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)  
        self.matcher_name = 'FlannFeatureMatcher'                                                

