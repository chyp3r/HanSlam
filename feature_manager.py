import math 
from enum import Enum
import numpy as np 
import cv2

from parameters import Parameters  

from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo

from utils_sys import Printer, import_from
from utils_features import unpackSiftOctaveKps, UnpackOctaveMethod, sat_num_features, kdt_nms, ssc_nms, octree_nms, grid_nms
from utils_geom import hamming_distance, hamming_distances, l2_distance, l2_distances

from feature_manager_adaptors import BlockAdaptor, PyramidAdaptor
from pyramid import PyramidType

from feature_root_sift import RootSIFTFeature2D
from feature_shitomasi import ShiTomasiDetector
    
kVerbose = True   

kNumFeatureDefault = Parameters.kNumFeatures

kNumLevelsDefault = 4
kScaleFactorDefault = 1.2 

kNumLevelsInitSigma = 40

kSigmaLevel0 = Parameters.kSigmaLevel0 

kDrawOriginalExtractedFeatures = False  

kFASTKeyPointSizeRescaleFactor = 4        
kAGASTKeyPointSizeRescaleFactor = 4       
kShiTomasiKeyPointSizeRescaleFactor = 5


if not kVerbose:
    def print(*args, **kwargs):
        pass 
    
    
class KeyPointFilterTypes(Enum):
    NONE         = 0
    SAT          = 1      
    KDT_NMS      = 2      
    SSC_NMS      = 3      
    OCTREE_NMS   = 4      
    GRID_NMS     = 5      


def feature_manager_factory(num_features=kNumFeatureDefault, 
                            num_levels = kNumLevelsDefault,                  
                            scale_factor = kScaleFactorDefault,             
                            detector_type = FeatureDetectorTypes.FAST, 
                            descriptor_type = FeatureDescriptorTypes.ORB):
    return FeatureManager(num_features, num_levels, scale_factor, detector_type, descriptor_type)

class FeatureManager(object):
    def __init__(self, num_features=kNumFeatureDefault, 
                       num_levels = kNumLevelsDefault,                        
                       scale_factor = kScaleFactorDefault,                     
                       detector_type = FeatureDetectorTypes.FAST,  
                       descriptor_type = FeatureDescriptorTypes.ORB):
        self.detector_type = detector_type 
        self._feature_detector   = None 
                
        self.descriptor_type = descriptor_type
        self._feature_descriptor = None 
                
        self.num_features = num_features
        self.num_levels = num_levels  
        self.first_level = 0              
                                         
        self.scale_factor = scale_factor  
        self.sigma_level0 = kSigmaLevel0 
        self.layers_per_octave = 3       
        
        self.norm_type = None            
        self.descriptor_distance = None  
        self.descriptor_distances = None  
                
        self.use_bock_adaptor = False 
        self.block_adaptor = None
        
        self.use_pyramid_adaptor = False 
        self.pyramid_adaptor = None 
        self.pyramid_type = PyramidType.RESIZE
        self.pyramid_do_parallel = True
        self.do_sat_features_per_level = False  
        self.force_multiscale_detect_and_compute = False 
        
        self.oriented_features = True            
        self.do_keypoints_size_rescaling = False 
        self.need_color_image = False             
                
        self.keypoint_filter_type = KeyPointFilterTypes.SAT           
        self.need_nms = False                                            
        self.keypoint_nms_filter_type = KeyPointFilterTypes.KDT_NMS   
        self.init_sigma_levels()
        

        print("using opencv ", cv2.__version__)
        opencv_major =  int(cv2.__version__.split('.')[0])
        opencv_minor =  int(cv2.__version__.split('.')[1])
        if opencv_major == 4 and opencv_minor >= 5: 
            SIFT_create  = import_from('cv2','SIFT_create') 
            ORB_create   = import_from('cv2','ORB_create')
            BRISK_create = import_from('cv2','BRISK_create')
            KAZE_create  = import_from('cv2','KAZE_create')            
            AKAZE_create = import_from('cv2','AKAZE_create')
                                  

        self.FAST_create  = import_from('cv2','FastFeatureDetector_create')
        self.AGAST_create = import_from('cv2','AgastFeatureDetector_create')       
        self.GFTT_create  = import_from('cv2','GFTTDetector_create')
        self.MSER_create  = import_from('cv2','MSER_create')
        self.SIFT_create = SIFT_create
        self.ORB_create = ORB_create 
        self.BRISK_create = BRISK_create            
        self.AKAZE_create = AKAZE_create   
        self.KAZE_create = KAZE_create           

        self.is_detector_equal_to_descriptor = (self.detector_type.name == self.descriptor_type.name)
                            
    
        if self.descriptor_type in [
                                    FeatureDescriptorTypes.SIFT,      
                                    FeatureDescriptorTypes.ROOT_SIFT,  
                                    FeatureDescriptorTypes.AKAZE,      
                                    FeatureDescriptorTypes.KAZE,                                          
                                    ]:
            self.scale_factor = 2 
            Printer.orange('forcing scale factor=2 for detector', self.descriptor_type.name)
            
        self.orb_params = dict(nfeatures=num_features,
                               scaleFactor=self.scale_factor,
                               nlevels=self.num_levels,
                               patchSize=31,
                               edgeThreshold = 10, 
                               fastThreshold = 20,
                               firstLevel = self.first_level,
                               WTA_K = 2,
                               scoreType=cv2.ORB_FAST_SCORE)  
        
        if self.detector_type == FeatureDetectorTypes.SIFT or self.detector_type == FeatureDetectorTypes.ROOT_SIFT:    
            sift = self.SIFT_create(nOctaveLayers=self.layers_per_octave)  
            self.set_sift_parameters()
            if self.detector_type == FeatureDetectorTypes.ROOT_SIFT:        
                self._feature_detector = RootSIFTFeature2D(sift)  
            else: 
                self._feature_detector = sift
            #
            #
        elif self.detector_type == FeatureDetectorTypes.SURF:          
            self._feature_detector = self.SURF_create(nOctaves = self.num_levels, nOctaveLayers=self.layers_per_octave)  
            self.scale_factor = 2                
            #
            #            
        elif self.detector_type == FeatureDetectorTypes.ORB:          
            self._feature_detector = self.ORB_create(**self.orb_params)               
            self.use_bock_adaptor = True          
            self.need_nms = self.num_levels > 1
            #
            #         
        elif self.detector_type == FeatureDetectorTypes.BRISK:          
            self._feature_detector = self.BRISK_create(octaves=self.num_levels) 
            self.scale_factor = 2                   
            #
            #     
        elif self.detector_type == FeatureDetectorTypes.KAZE:           
            self._feature_detector = self.KAZE_create(nOctaves=self.num_levels, threshold=0.0005)  
            self.scale_factor = 2                 
            #
            #                          
        elif self.detector_type == FeatureDetectorTypes.AKAZE:           
            self._feature_detector = self.AKAZE_create(nOctaves=self.num_levels, threshold=0.0005) 
            self.scale_factor = 2                   
            #
            #                                                                                
        elif self.detector_type == FeatureDetectorTypes.FAST:    
            self.oriented_features = False             
            self._feature_detector = self.FAST_create(threshold=20, nonmaxSuppression=True)   
            if self.descriptor_type != FeatureDescriptorTypes.NONE:                   
                self.use_pyramid_adaptor = self.num_levels > 1                           
                self.need_nms = self.num_levels > 1   
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS     
                self.do_keypoints_size_rescaling = True             
            #  
            #  
        elif self.detector_type == FeatureDetectorTypes.SHI_TOMASI:         
            self.oriented_features = False                       
            self._feature_detector = ShiTomasiDetector(self.num_features)  
            if self.descriptor_type != FeatureDescriptorTypes.NONE:            
                self.use_pyramid_adaptor = self.num_levels > 1 
                self.need_nms = self.num_levels > 1   
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS          
                self.do_keypoints_size_rescaling = True     
            #    
            #      
        elif self.detector_type == FeatureDetectorTypes.AGAST:  
            self.oriented_features = False               
            self._feature_detector = self.AGAST_create(threshold=10, nonmaxSuppression=True)    
            if self.descriptor_type != FeatureDescriptorTypes.NONE:             
                self.use_pyramid_adaptor = self.num_levels > 1              
                self.need_nms = self.num_levels > 1   
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS
                self.do_keypoints_size_rescaling = True                 
            # 
            #       
        elif self.detector_type == FeatureDetectorTypes.GFTT:    
            self.oriented_features = False 
            self._feature_detector = self.GFTT_create(self.num_features, qualityLevel=0.01, minDistance=3, blockSize=5, useHarrisDetector=False, k=0.04)    
            if self.descriptor_type != FeatureDescriptorTypes.NONE:              
                self.use_pyramid_adaptor = self.num_levels > 1                   
                self.need_nms = self.num_levels > 1   
                self.keypoint_nms_filter_type = KeyPointFilterTypes.OCTREE_NMS      
                self.do_keypoints_size_rescaling = True           
            #   
            # 
        elif self.detector_type == FeatureDetectorTypes.MSER:    
            self._feature_detector = self.MSER_create()    
            self.use_pyramid_adaptor = self.num_levels > 1  
            self.pyramid_do_parallel = False      
            self.need_nms = self.num_levels > 1   
            #     
            # 
        elif self.detector_type == FeatureDetectorTypes.MSD:    
            self._feature_detector = self.MSD_create()   
            print('MSD detector info:',dir(self._feature_detector))           
            #      
            #      
        elif self.detector_type == FeatureDetectorTypes.STAR:  
            self.oriented_features = False               
            self._feature_detector = self.STAR_create(maxSize=45,
                                                      responseThreshold=10, 
                                                      lineThresholdProjected=10,
                                                      lineThresholdBinarized=8,
                                                      suppressNonmaxSize=5)  
            if self.descriptor_type != FeatureDescriptorTypes.NONE:              
                self.use_pyramid_adaptor = self.num_levels > 1   
            #   
            #  
        elif self.detector_type == FeatureDetectorTypes.HL:  
            self.oriented_features = False               
            self._feature_detector = self.HL_create(numOctaves=self.num_levels,
                                                    corn_thresh=0.005,
                                                    DOG_thresh=0.01,  
                                                    maxCorners=self.num_features, 
                                                    num_layers=4)  
            self.scale_factor = 2   
            #           
            #                                                                                                                              
        else:
            raise ValueError("Unknown feature detector %s" % self.detector_type)
                
        if self.need_nms:
            self.keypoint_filter_type = self.keypoint_nms_filter_type         
        
        if self.use_bock_adaptor: 
              self.orb_params['edgeThreshold'] = 0
                                     
        if self.is_detector_equal_to_descriptor:     
            Printer.green('using same detector and descriptor object: ', self.detector_type.name)
            self._feature_descriptor = self._feature_detector
        else:      
            self.num_levels_descriptor = self.num_levels                    
            if self.use_pyramid_adaptor:        
                pass 
            if self.descriptor_type == FeatureDescriptorTypes.SIFT or self.descriptor_type == FeatureDescriptorTypes.ROOT_SIFT:                      
                sift = self.SIFT_create(nOctaveLayers=3)   
                if self.descriptor_type == FeatureDescriptorTypes.ROOT_SIFT:            
                    self._feature_descriptor = RootSIFTFeature2D(sift)         
                else: 
                    self._feature_descriptor = sift
                #
                #        
            elif self.descriptor_type == FeatureDescriptorTypes.SURF:       
                self.oriented_features = True                        
                self._feature_descriptor = self.SURF_create(nOctaves = self.num_levels_descriptor, nOctaveLayers=3)      
                #
                #   
            elif self.descriptor_type == FeatureDescriptorTypes.ORB:             
                self._feature_descriptor = self.ORB_create(**self.orb_params) 
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.ORB2: 
                self._feature_descriptor = self.ORB_create(**self.orb_params)                           
                #
                #                      
            elif self.descriptor_type == FeatureDescriptorTypes.BRISK:    
                self.oriented_features = True                          
                self._feature_descriptor = self.BRISK_create(octaves=self.num_levels_descriptor)        
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.KAZE:     
                if not self.is_detector_equal_to_descriptor:
                    Printer.red('WARNING: KAZE descriptors can only be used with KAZE or AKAZE keypoints.')  
                self._feature_descriptor = self.KAZE_create(nOctaves=self.num_levels_descriptor) 
                #
                #                
            elif self.descriptor_type == FeatureDescriptorTypes.AKAZE:     
                if not self.is_detector_equal_to_descriptor:
                    Printer.red('WARNING: AKAZE descriptors can only be used with KAZE or AKAZE keypoints.')  
                self._feature_descriptor = self.AKAZE_create(nOctaves=self.num_levels_descriptor) 
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.FREAK: 
                self.oriented_features = True                         
                self._feature_descriptor = self.FREAK_create(nOctaves=self.num_levels_descriptor)   
                #
                #
            elif self.descriptor_type == FeatureDescriptorTypes.SUPERPOINT:              
                if self.detector_type != FeatureDetectorTypes.SUPERPOINT: 
                    raise ValueError("You cannot use SUPERPOINT descriptor without SUPERPOINT detector!\nPlease, select SUPERPOINT as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                             
                #
                #      
            elif self.descriptor_type == FeatureDescriptorTypes.BOOST_DESC:
                self.do_keypoints_size_rescaling = False 
                boost_des_keypoint_size_scale_factor = 1.5   
                if self.detector_type in [FeatureDetectorTypes.KAZE, FeatureDetectorTypes.SURF]:
                    boost_des_keypoint_size_scale_factor = 6.25
                elif self.detector_type == FeatureDetectorTypes.SIFT:
                    boost_des_keypoint_size_scale_factor = 6.75         
                elif self.detector_type in [FeatureDetectorTypes.AKAZE, FeatureDetectorTypes.AGAST, FeatureDetectorTypes.FAST, FeatureDetectorTypes.BRISK]:
                    boost_des_keypoint_size_scale_factor = 5.0    
                elif self.detector_type == FeatureDetectorTypes.ORB:
                    boost_des_keypoint_size_scale_factor = 0.75                                     
                self._feature_descriptor = self.BoostDesc_create(scale_factor=boost_des_keypoint_size_scale_factor)         
                #
                #   
            elif self.descriptor_type == FeatureDescriptorTypes.DAISY:              
                self._feature_descriptor = self.DAISY_create()        
                #
                #             
            elif self.descriptor_type == FeatureDescriptorTypes.LATCH:              
                self._feature_descriptor = self.LATCH_create()        
                #
                #       
            elif self.descriptor_type == FeatureDescriptorTypes.LUCID:              
                self._feature_descriptor = self.LUCID_create(lucid_kernel=1,  
                                                             blur_kernel=3 )       
                self.need_color_image = True 
                #
                #      
            elif self.descriptor_type == FeatureDescriptorTypes.VGG:              
                self._feature_descriptor = self.VGG_create()        
                #
                #       
            elif self.descriptor_type == FeatureDescriptorTypes.D2NET:   
                self.need_color_image = True           
                if self.detector_type != FeatureDetectorTypes.D2NET: 
                    raise ValueError("You cannot use D2NET descriptor without D2NET detector!\nPlease, select D2NET as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                                      
                #
                #            
            elif self.descriptor_type == FeatureDescriptorTypes.DELF:   
                self.need_color_image = True           
                if self.detector_type != FeatureDetectorTypes.DELF: 
                    raise ValueError("You cannot use DELF descriptor without DELF detector!\nPlease, select DELF as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                                      
                #
                #      
            elif self.descriptor_type == FeatureDescriptorTypes.CONTEXTDESC:   
                self.need_color_image = True           
                if self.detector_type != FeatureDetectorTypes.CONTEXTDESC: 
                    raise ValueError("You cannot use CONTEXTDESC descriptor without CONTEXTDESC detector!\nPlease, select CONTEXTDESC as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                                    
                #
                #         
            elif self.descriptor_type == FeatureDescriptorTypes.LFNET:   
                self.need_color_image = True           
                if self.detector_type != FeatureDetectorTypes.LFNET: 
                    raise ValueError("You cannot use LFNET descriptor without LFNET detector!\nPlease, select LFNET as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                                  
                #
                #    
            elif self.descriptor_type == FeatureDescriptorTypes.R2D2:   
                self.oriented_features = False                     
                self.need_color_image = True           
                if self.detector_type != FeatureDetectorTypes.R2D2: 
                    raise ValueError("You cannot use R2D2 descriptor without R2D2 detector!\nPlease, select R2D2 as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                                    
                #
                #     
            elif self.descriptor_type == FeatureDescriptorTypes.KEYNET:   
                self.oriented_features = False                           
                if self.detector_type != FeatureDetectorTypes.KEYNET: 
                    raise ValueError("You cannot use KEYNET internal descriptor without KEYNET detector!\nPlease, select KEYNET as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                                   
                #
                #          
            elif self.descriptor_type == FeatureDescriptorTypes.BEBLID:     
                BEBLID_SIZE_256_BITS = 101   
                BEBLID_scale_factor = 1.0  
                self._feature_descriptor = self.BEBLID_create(BEBLID_scale_factor, BEBLID_SIZE_256_BITS)        
                #
                #     
            elif self.descriptor_type == FeatureDescriptorTypes.DISK:   
                self.oriented_features = False                           
                if self.detector_type != FeatureDetectorTypes.DISK: 
                    raise ValueError("You cannot use DISK internal descriptor without DISK detector!\nPlease, select DISK as both descriptor and detector!")
                self._feature_descriptor = self._feature_detector                                     
                #
                #                                                                                                                                                                                                                                                               
            elif self.descriptor_type == FeatureDescriptorTypes.NONE:        
                self._feature_descriptor = None                                              
            else:
                raise ValueError("Unknown feature descriptor %s" % self.descriptor_type)    
            
        try: 
            self.norm_type = FeatureInfo.norm_type[self.descriptor_type]
        except:
            Printer.red('You did not set the norm type for: ', self.descriptor_type.name)              
            raise ValueError("Unmanaged norm type for feature descriptor %s" % self.descriptor_type.name)     
        
        if self.norm_type == cv2.NORM_HAMMING:
            self.descriptor_distance = hamming_distance
            self.descriptor_distances = hamming_distances            
        if self.norm_type == cv2.NORM_L2:
            self.descriptor_distance = l2_distance      
            self.descriptor_distances = l2_distances         
            
        try: 
            Parameters.kMaxDescriptorDistance = FeatureInfo.max_descriptor_distance[self.descriptor_type]
        except: 
            Printer.red('You did not set the reference max descriptor distance for: ', self.descriptor_type.name)                                                         
            raise ValueError("Unmanaged max descriptor distance for feature descriptor %s" % self.descriptor_type.name)                
        Parameters.kMaxDescriptorDistanceSearchEpipolar = Parameters.kMaxDescriptorDistance                    
                
        if not self.oriented_features:
            Printer.orange('WARNING: using NON-ORIENTED features: ', self.detector_type.name,'-',self.descriptor_type.name, ' (i.e. kp.angle=0)')     
                        
        if self.is_detector_equal_to_descriptor and \
            ( self.detector_type == FeatureDetectorTypes.SIFT or 
              self.detector_type == FeatureDetectorTypes.ROOT_SIFT or 
              self.detector_type == FeatureDetectorTypes.CONTEXTDESC ):        
            self.init_sigma_levels_sift()                 
        else: 
            self.init_sigma_levels()    
            
        if self.use_bock_adaptor:
            self.block_adaptor = BlockAdaptor(self._feature_detector, self._feature_descriptor)

        if self.use_pyramid_adaptor:   
            self.pyramid_params = dict(detector=self._feature_detector, 
                                       descriptor=self._feature_descriptor, 
                                       num_features = self.num_features,
                                       num_levels=self.num_levels, 
                                       scale_factor=self.scale_factor, 
                                       sigma0=self.sigma_level0, 
                                       first_level=self.first_level, 
                                       pyramid_type=self.pyramid_type,
                                       use_block_adaptor=self.use_bock_adaptor,
                                       do_parallel = self.pyramid_do_parallel,
                                       do_sat_features_per_level = self.do_sat_features_per_level)       
            self.pyramid_adaptor = PyramidAdaptor(**self.pyramid_params)
         
    
    def set_sift_parameters(self):
        self.scale_factor = 2                
        self.sigma_level0 = 1.6            
        self.first_level = -1           

    def init_sigma_levels(self): 
        print('num_levels: ', self.num_levels)               
        num_levels = max(kNumLevelsInitSigma, self.num_levels)    
        self.inv_scale_factor = 1./self.scale_factor      
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.level_sigmas = np.zeros(num_levels)                
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)
        self.log_scale_factor = math.log(self.scale_factor)

        self.scale_factors[0] = 1.0                   
        self.level_sigmas2[0] = self.sigma_level0*self.sigma_level0
        self.level_sigmas[0] = math.sqrt(self.level_sigmas2[0])
        for i in range(1,num_levels):
            self.scale_factors[i] = self.scale_factors[i-1]*self.scale_factor
            self.level_sigmas2[i] = self.scale_factors[i]*self.scale_factors[i]*self.level_sigmas2[0]  
            self.level_sigmas[i]  = math.sqrt(self.level_sigmas2[i])        
        for i in range(num_levels):
            self.inv_scale_factors[i] = 1.0/self.scale_factors[i]
            self.inv_level_sigmas2[i] = 1.0/self.level_sigmas2[i]

    def init_sigma_levels_sift(self): 
        print('initializing SIFT sigma levels')
        print('num_levels: ', self.num_levels)          
        self.num_levels = 3*self.num_levels + 3   
        num_levels = max(kNumLevelsInitSigma, self.num_levels) 
        print('original scale factor: ', self.scale_factor)
        self.scale_factor = math.pow(2,1./3)    
        self.inv_scale_factor = 1./self.scale_factor      
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.level_sigmas = np.zeros(num_levels)                
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)
        self.log_scale_factor = math.log(self.scale_factor)
        
        self.sigma_level0 = 1.6 
        sigma_level02 = self.sigma_level0*self.sigma_level0        

        self.scale_factors[0] = 1.0   
        self.level_sigmas2[0] = sigma_level02 
                                                                              
        self.level_sigmas[0] = math.sqrt(self.level_sigmas2[0])
        for i in range(1,num_levels):
            self.scale_factors[i] = self.scale_factors[i-1]*self.scale_factor
            self.level_sigmas2[i] = self.scale_factors[i]*self.scale_factors[i]*sigma_level02  
            self.level_sigmas[i]  = math.sqrt(self.level_sigmas2[i])         
        for i in range(num_levels):
            self.inv_scale_factors[i] = 1.0/self.scale_factors[i]
            self.inv_level_sigmas2[i] = 1.0/self.level_sigmas2[i]

    def filter_keypoints(self, type, frame, kps, des=None):
        filter_name = type.name        
        if type == KeyPointFilterTypes.NONE:
            pass  
        elif type == KeyPointFilterTypes.KDT_NMS:      
            kps, des = kdt_nms(kps, des, self.num_features)
        elif type == KeyPointFilterTypes.SSC_NMS:    
            kps, des = ssc_nms(kps, des, frame.shape[1], frame.shape[0], self.num_features)   
        elif type == KeyPointFilterTypes.OCTREE_NMS:
            if des is not None: 
                raise ValueError('at the present time, you cannot use OCTREE_NMS with descriptors')
            kps = octree_nms(frame, kps, self.num_features)
        elif type == KeyPointFilterTypes.GRID_NMS:    
            kps, des, _ = grid_nms(kps, des, frame.shape[0], frame.shape[1], self.num_features, dist_thresh=4)            
        elif type == KeyPointFilterTypes.SAT:                                                        
            if len(kps) > self.num_features:
                kps, des = sat_num_features(kps, des, self.num_features)      
        else:             
            raise ValueError("Unknown match-filter type")     
        return kps, des, filter_name 
             

    def rescale_keypoint_size(self, kps):    
        scale = 1   
        doit = False 
        if self.detector_type == FeatureDetectorTypes.FAST:
            scale = kFASTKeyPointSizeRescaleFactor
            doit = True 
        elif self.detector_type == FeatureDetectorTypes.AGAST:
            scale = kAGASTKeyPointSizeRescaleFactor     
            doit = True                    
        elif self.detector_type == FeatureDetectorTypes.SHI_TOMASI or self.detector_type == FeatureDetectorTypes.GFTT:
            scale = kShiTomasiKeyPointSizeRescaleFactor
            doit = True             
        if doit: 
            for kp in kps:
                kp.size *= scale     
                             
    def detect(self, frame, mask=None, filter=True): 
        if not self.need_color_image and frame.ndim>2:                                   
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)                    
        if self.use_pyramid_adaptor:  
            kps = self.pyramid_adaptor.detect(frame, mask)            
        elif self.use_bock_adaptor:   
            kps = self.block_adaptor.detect(frame, mask)            
        else:                         
            kps = self._feature_detector.detect(frame, mask)  
        filter_name = 'NONE'   
        if filter:   
            kps, _, filter_name  = self.filter_keypoints(self.keypoint_filter_type, frame, kps) 
        if self.do_keypoints_size_rescaling:
            self.rescale_keypoint_size(kps)             
        if kDrawOriginalExtractedFeatures: 
            imgDraw = cv2.drawKeypoints(frame, kps, None, color=(0,255,0), flags=0)
            cv2.imshow('detected keypoints',imgDraw)            
        if kVerbose:
            print('detector:',self.detector_type.name,', #features:', len(kps),', [kp-filter:',filter_name,']')    
        return kps        
    
    def compute(self, frame, kps, filter = True):
        if not self.need_color_image and frame.ndim>2:    
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)          
        kps, des = self._feature_descriptor.compute(frame, kps)  
        filter_name = 'NONE'                 
        if filter: 
            kps, des, filter_name  = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)            
        if kVerbose:
            print('descriptor:',self.descriptor_type.name,', #features:', len(kps),', [kp-filter:',filter_name,']')           
        return kps, des 

    def detectAndCompute(self, frame, mask=None, filter = True):
        if not self.need_color_image and frame.ndim>2:    
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)  
        if self.use_pyramid_adaptor:  
            if self.force_multiscale_detect_and_compute: 
                kps, des = self.pyramid_adaptor.detectAndCompute(frame, mask)  
            
            else: 
                kps = self.detect(frame, mask, filter=True)                                  
                kps, des = self.compute(frame, kps, filter=False) 
                filter = False
        elif self.use_bock_adaptor:   
            kps = self.detect(frame, mask, filter=True)                              
            kps, des = self.compute(frame, kps, filter=False)   
            filter = False              
        else:                         
            if self.is_detector_equal_to_descriptor:                     
                kps, des = self._feature_detector.detectAndCompute(frame, mask)   
                if kVerbose:
                    print('detector:', self.detector_type.name,', #features:',len(kps))           
                    print('descriptor:', self.descriptor_type.name,', #features:',len(kps))                      
            else:
                kps = self.detect(frame, mask, filter=False)                  
                kps, des = self._feature_descriptor.compute(frame, kps)  
                if kVerbose:
                    print('descriptor: ', self.descriptor_type.name, ', #features: ', len(kps))   
        filter_name = 'NONE'
        if filter:                                                                 
            kps, des, filter_name  = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)                                                              
        if self.detector_type == FeatureDetectorTypes.SIFT or \
           self.detector_type == FeatureDetectorTypes.ROOT_SIFT or \
           self.detector_type == FeatureDetectorTypes.CONTEXTDESC :
            unpackSiftOctaveKps(kps, method=UnpackOctaveMethod.INTRAL_LAYERS)           
        if kVerbose:
            print('detector:',self.detector_type.name,', descriptor:', self.descriptor_type.name,', #features:', len(kps),' (#ref:', self.num_features, '), [kp-filter:',filter_name,']')                                         
        return kps, des             
 
                    
