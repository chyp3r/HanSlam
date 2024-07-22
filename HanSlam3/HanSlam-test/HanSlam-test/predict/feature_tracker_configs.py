from .feature_tracker import FeatureTrackerTypes 
from .feature_types import FeatureDetectorTypes, FeatureDescriptorTypes
from .parameters import Parameters  

kNumFeatures=Parameters.kNumFeatures    

kRatioTest=Parameters.kFeatureMatchRatioTest

kTrackerType = FeatureTrackerTypes.DES_BF     
        

class FeatureTrackerConfigs(object):   
    TEST = dict(num_features=kNumFeatures,                   
                num_levels = 8,                                 
                scale_factor = 1.2,                              
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.ORB2, 
                match_ratio_test = kRatioTest,
                tracker_type = kTrackerType)
    
    LK_SHI_TOMASI = dict(num_features=kNumFeatures,
                         num_levels = 3,
                         detector_type = FeatureDetectorTypes.SHI_TOMASI,
                         descriptor_type = FeatureDescriptorTypes.NONE, 
                         tracker_type = FeatureTrackerTypes.LK)

    LK_FAST = dict(num_features=kNumFeatures,
                   num_levels = 3,
                   detector_type = FeatureDetectorTypes.FAST, 
                   descriptor_type = FeatureDescriptorTypes.NONE, 
                   tracker_type = FeatureTrackerTypes.LK)

    SHI_TOMASI_ORB = dict(num_features=kNumFeatures,                  
                          num_levels = 8, 
                          scale_factor = 1.2,
                          detector_type = FeatureDetectorTypes.SHI_TOMASI, 
                          descriptor_type = FeatureDescriptorTypes.ORB, 
                          match_ratio_test = kRatioTest,
                          tracker_type = kTrackerType)
    
    SHI_TOMASI_FREAK = dict(num_features=kNumFeatures,                     
                            num_levels=8,                      
                            scale_factor = 1.2,
                            detector_type = FeatureDetectorTypes.SHI_TOMASI, 
                            descriptor_type = FeatureDescriptorTypes.FREAK, 
                            match_ratio_test = kRatioTest,
                            tracker_type = kTrackerType)      

    FAST_ORB = dict(num_features=kNumFeatures,                        
                    num_levels = 8, 
                    scale_factor = 1.2,
                    detector_type = FeatureDetectorTypes.FAST, 
                    descriptor_type = FeatureDescriptorTypes.ORB, 
                    match_ratio_test = kRatioTest,                         
                    tracker_type = kTrackerType) 
    
    FAST_FREAK = dict(num_features=kNumFeatures,                       
                      num_levels = 8,
                      scale_factor = 1.2,                    
                      detector_type = FeatureDetectorTypes.FAST, 
                      descriptor_type = FeatureDescriptorTypes.FREAK,      
                      match_ratio_test = kRatioTest,                          
                      tracker_type = kTrackerType)       

    BRISK = dict(num_features=kNumFeatures,                     
                num_levels = 4, 
                scale_factor = 1.2,
                detector_type = FeatureDetectorTypes.BRISK, 
                descriptor_type = FeatureDescriptorTypes.BRISK, 
                match_ratio_test = kRatioTest,                               
                tracker_type = kTrackerType)  
    
    BRISK_TFEAT = dict(num_features=kNumFeatures,                     
                       num_levels = 4, 
                       scale_factor = 1.2,
                       detector_type = FeatureDetectorTypes.BRISK, 
                       descriptor_type = FeatureDescriptorTypes.TFEAT, 
                       match_ratio_test = kRatioTest,                               
                       tracker_type = kTrackerType)        

    ORB = dict(num_features=kNumFeatures, 
               num_levels = 8, 
               scale_factor = 1.2, 
               detector_type = FeatureDetectorTypes.ORB, 
               descriptor_type = FeatureDescriptorTypes.ORB, 
               match_ratio_test = kRatioTest,                        
               tracker_type = kTrackerType)
    
    ORB2 = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.ORB2, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)    
    
    BRISK = dict(num_features=kNumFeatures,
                 num_levels = 8,
                 detector_type = FeatureDetectorTypes.BRISK, 
                 descriptor_type = FeatureDescriptorTypes.BRISK,
                 match_ratio_test = kRatioTest,                           
                 tracker_type = kTrackerType)   

    KAZE = dict(num_features=kNumFeatures,
                num_levels = 8,
                detector_type = FeatureDetectorTypes.KAZE, 
                descriptor_type = FeatureDescriptorTypes.KAZE, 
                match_ratio_test = kRatioTest,                          
                tracker_type = kTrackerType)  
    
    AKAZE = dict(num_features=kNumFeatures,
                 num_levels = 8,
                 detector_type = FeatureDetectorTypes.AKAZE, 
                 descriptor_type = FeatureDescriptorTypes.AKAZE, 
                 match_ratio_test = kRatioTest,                          
                 tracker_type = kTrackerType)  
                
    SIFT = dict(num_features=kNumFeatures,
                detector_type = FeatureDetectorTypes.SIFT, 
                descriptor_type = FeatureDescriptorTypes.SIFT, 
                match_ratio_test = kRatioTest,                         
                tracker_type = kTrackerType)
    
    ROOT_SIFT = dict(num_features=kNumFeatures,
                     detector_type = FeatureDetectorTypes.ROOT_SIFT, 
                     descriptor_type = FeatureDescriptorTypes.ROOT_SIFT, 
                     match_ratio_test = kRatioTest,                              
                     tracker_type = kTrackerType)    
    
    SURF = dict(num_features=kNumFeatures,
                num_levels = 8,
                detector_type = FeatureDetectorTypes.SURF, 
                descriptor_type = FeatureDescriptorTypes.SURF, 
                match_ratio_test = kRatioTest,                         
                tracker_type = kTrackerType)
        
    SUPERPOINT = dict(num_features=kNumFeatures,                          
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.SUPERPOINT, 
                      descriptor_type = FeatureDescriptorTypes.SUPERPOINT, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = kTrackerType)

    CONTEXTDESC = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.CONTEXTDESC, 
                       descriptor_type = FeatureDescriptorTypes.CONTEXTDESC, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    
    KEYNET = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.KEYNET, 
                       descriptor_type = FeatureDescriptorTypes.KEYNET, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
        
    DISK = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.DISK, 
                       descriptor_type = FeatureDescriptorTypes.DISK, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    
    ORB2_FREAK = dict(num_features=kNumFeatures, 
                      num_levels = 8, 
                      scale_factor = 1.2,                     
                      detector_type = FeatureDetectorTypes.ORB2, 
                      descriptor_type = FeatureDescriptorTypes.FREAK, 
                      match_ratio_test = kRatioTest,                        
                      tracker_type = kTrackerType)    
    
    ORB2_BEBLID = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.BEBLID, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)    
    
    ORB2_HARDNET = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.HARDNET, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)    
    
    ORB2_SOSNET = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.SOSNET, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)   
    
    ORB2_L2NET = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.L2NET, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType) 
