import numpy as np 
import cv2
from enum import Enum

from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from utils_sys import Printer, import_from
from utils_geom import hamming_distance, hamming_distances, l2_distance, l2_distances
from parameters import Parameters 


kMinNumFeatureDefault = 2000
kLkPyrOpticFlowNumLevelsMin = 3   
kRatioTest = Parameters.kFeatureMatchRatioTest


class FeatureTrackerTypes(Enum):
    LK        = 0   
    DES_BF    = 1   
    DES_FLANN = 2    


def feature_tracker_factory(num_features=kMinNumFeatureDefault, 
                            num_levels = 1,                                
                            scale_factor = 1.2,                            
                            detector_type = FeatureDetectorTypes.FAST, 
                            descriptor_type = FeatureDescriptorTypes.ORB, 
                            match_ratio_test = kRatioTest,
                            tracker_type = FeatureTrackerTypes.LK):
    if tracker_type == FeatureTrackerTypes.LK:
        return LkFeatureTracker(num_features=num_features, 
                                num_levels = num_levels, 
                                scale_factor = scale_factor, 
                                detector_type = detector_type, 
                                descriptor_type = descriptor_type, 
                                match_ratio_test = match_ratio_test,                                
                                tracker_type = tracker_type)
    else: 
        return DescriptorFeatureTracker(num_features=num_features, 
                                        num_levels = num_levels, 
                                        scale_factor = scale_factor, 
                                        detector_type = detector_type, 
                                        descriptor_type = descriptor_type,
                                        match_ratio_test = match_ratio_test,    
                                        tracker_type = tracker_type)
    return None 


class FeatureTrackingResult(object): 
    def __init__(self):
        self.kps_ref = None         
        self.kps_cur = None          
        self.des_cur = None         
        self.idxs_ref = None         
        self.idxs_cur = None         
        self.kps_ref_matched = None  
        self.kps_cur_matched = None  


class FeatureTracker(object): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 1,                                   
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.ORB,
                       match_ratio_test = kRatioTest, 
                       tracker_type = FeatureTrackerTypes.LK):
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.tracker_type = tracker_type

        self.feature_manager = None      
        self.matcher = None              
                
    @property
    def num_features(self):
        return self.feature_manager.num_features
    
    @property
    def num_levels(self):
        return self.feature_manager.num_levels    
    
    @property
    def scale_factor(self):
        return self.feature_manager.scale_factor    
    
    @property
    def norm_type(self):
        return self.feature_manager.norm_type       
    
    @property
    def descriptor_distance(self):
        return self.feature_manager.descriptor_distance               
    
    @property
    def descriptor_distances(self):
        return self.feature_manager.descriptor_distances               
    
    # out: keypoints and descriptors 
    def detectAndCompute(self, frame, mask): 
        return None, None 

    # out: FeatureTrackingResult()
    def track(self, image_ref, image_cur, kps_ref, des_ref):
        return FeatureTrackingResult()             


class LkFeatureTracker(FeatureTracker): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 3,                            
                       scale_factor = 1.2,                       
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.NONE, 
                       match_ratio_test = kRatioTest,
                       tracker_type = FeatureTrackerTypes.LK):                         
        super().__init__(num_features=num_features, 
                         num_levels=num_levels, 
                         scale_factor=scale_factor, 
                         detector_type=detector_type, 
                         descriptor_type=descriptor_type, 
                         tracker_type=tracker_type)
        self.feature_manager = feature_manager_factory(num_features=num_features, 
                                                       num_levels=num_levels, 
                                                       scale_factor=scale_factor, 
                                                       detector_type=detector_type, 
                                                       descriptor_type=descriptor_type)   
    
        optic_flow_num_levels = max(kLkPyrOpticFlowNumLevelsMin,num_levels)
        Printer.green('LkFeatureTracker: num levels on LK pyr optic flow: ', optic_flow_num_levels)
        self.lk_params = dict(winSize  = (21, 21), 
                              maxLevel = optic_flow_num_levels,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))        

    def detectAndCompute(self, frame, mask=None):
        return self.feature_manager.detect(frame, mask), None  

    def track(self, image_ref, image_cur, kps_ref, des_ref = None):
        kps_cur, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, kps_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        res = FeatureTrackingResult()    
        res.idxs_ref = [i for i,v in enumerate(st) if v== 1]
        res.idxs_cur = res.idxs_ref.copy()       
        res.kps_ref_matched = kps_ref[res.idxs_ref] 
        res.kps_cur_matched = kps_cur[res.idxs_cur]  
        res.kps_ref = res.kps_ref_matched  
        res.kps_cur = res.kps_cur_matched
        res.des_cur = None                      
        return res         
        

class DescriptorFeatureTracker(FeatureTracker): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 1,                                   
                       scale_factor = 1.2,                                    
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.ORB,
                       match_ratio_test = kRatioTest, 
                       tracker_type = FeatureTrackerTypes.DES_FLANN):
        super().__init__(num_features=num_features, 
                         num_levels=num_levels, 
                         scale_factor=scale_factor, 
                         detector_type=detector_type, 
                         descriptor_type=descriptor_type, 
                         match_ratio_test = match_ratio_test,
                         tracker_type=tracker_type)
        self.feature_manager = feature_manager_factory(num_features=num_features, 
                                                       num_levels=num_levels, 
                                                       scale_factor=scale_factor, 
                                                       detector_type=detector_type, 
                                                       descriptor_type=descriptor_type)                     

        if tracker_type == FeatureTrackerTypes.DES_FLANN:
            self.matching_algo = FeatureMatcherTypes.FLANN
        elif tracker_type == FeatureTrackerTypes.DES_BF:
            self.matching_algo = FeatureMatcherTypes.BF
        else:
            raise ValueError("Unmanaged matching algo for feature tracker %s" % self.tracker_type)                   
                    
        self.matcher = feature_matcher_factory(norm_type=self.norm_type, ratio_test=match_ratio_test, type=self.matching_algo)        

    def detectAndCompute(self, frame, mask=None):
        return self.feature_manager.detectAndCompute(frame, mask) 

    def track(self, image_ref, image_cur, kps_ref, des_ref):
        kps_cur, des_cur = self.detectAndCompute(image_cur)
        kps_cur = np.array([x.pt for x in kps_cur], dtype=np.float32) 
    
        idxs_ref, idxs_cur = self.matcher.match(des_ref, des_cur)  

        res = FeatureTrackingResult()
        res.kps_ref = kps_ref 
        res.kps_cur = kps_cur 
        res.des_cur = des_cur
        
        res.kps_ref_matched = np.asarray(kps_ref[idxs_ref])
        res.idxs_ref = np.asarray(idxs_ref)                  
        
        res.kps_cur_matched = np.asarray(kps_cur[idxs_cur]) 
        res.idxs_cur = np.asarray(idxs_cur)
        
        return res                 
