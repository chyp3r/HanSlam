import math
import numpy as np 
from concurrent.futures import ThreadPoolExecutor, wait
from .utils_img import img_mask_blocks
from .utils_features import sat_num_features
from .pyramid import Pyramid, PyramidType
kVerbose = True   

kAdaptorNumRowDivs = 5 
kAdaptorNumColDivs = 5 

kNumLevelsInitSigma = 20

kBlockAdaptorUseParallelComputations = True 
kBlockAdaptorMaxNumWorkers = 4

kPyramidAdaptorUseParallelComputations = True 
kPyramidAdaptorMaxNumWorkers = 4


if not kVerbose:
    def print(*args, **kwargs):
        pass 

class BlockAdaptor(object): 
    def __init__(self, 
                 detector, 
                 descriptor = None, 
                 row_divs = kAdaptorNumRowDivs, 
                 col_divs = kAdaptorNumColDivs,
                 do_parallel = kBlockAdaptorUseParallelComputations):    
        self.detector = detector 
        self.descriptor = descriptor 
        self.row_divs = row_divs
        self.col_divs = col_divs 
        self.do_parallel = do_parallel  
        self.is_detector_equal_to_descriptor = (self.detector == self.descriptor)


    def detect(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1: 
            return self.detector.detect(frame, mask)
        else:   
            block_generator = img_mask_blocks(frame, mask, self.row_divs, self.col_divs)
            kps_all = []   
            
            def detect_block(b_m_i_j):                         
                b, m, i, j = b_m_i_j               
                kps = self.detector.detect(b, mask=m)
                for kp in kps:
                    kp.pt = (kp.pt[0] + j, kp.pt[1] + i)        
                kps_all.extend(kps)     
                       
            if not self.do_parallel:
                for b, m, i, j in block_generator:
                    detect_block((b,m,i,j))
            else: 
                with ThreadPoolExecutor(max_workers = 4) as executor:
                    executor.map(detect_block, block_generator)                
            return np.array(kps_all)
        
    def detectAndCompute(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1: 
            return self.detector.detectAndCompute(frame, mask)
        else:   
            block_generator = img_mask_blocks(frame, mask, self.row_divs, self.col_divs)
            kps_all = []
            des_all = []
            kps_des_map = {} 
            
            def detect_and_compute_block(b_m_i_j):                         
                b, m, i, j = b_m_i_j
                if self.is_detector_equal_to_descriptor:             
                    kps, des = self.detector.detectAndCompute(b, mask=m)
                else:
                    kps = self.detector.detect(b, mask=m)    
                    kps, des = self.descriptor.compute(b, kps)  

                for kp in kps:
                    kp.pt = (kp.pt[0] + j, kp.pt[1] + i)        
                kps_des_map[(i,j)] = (kps,des)     
                             
            if not self.do_parallel:
                for b, m, i, j in block_generator:
                    detect_and_compute_block((b, m, i, j))  
            else: 
                with ThreadPoolExecutor(max_workers = kBlockAdaptorMaxNumWorkers) as executor:
                    executor.map(detect_and_compute_block, block_generator)
                    
            for ij,(kps,des) in kps_des_map.items():
                kps_all.extend(kps)       
                if des is not None and len(des)>0:                    
                    if len(des_all)>0:
                        des_all = np.vstack([des_all, des])                                       
                    else:                  
                        des_all = des                                                                                                
            return np.array(kps_all), np.array(des_all)     

class PyramidAdaptor(object): 
    def __init__(self, 
                 detector, 
                 descriptor=None,
                 num_features=2000, 
                 num_levels=4, 
                 scale_factor=1.2, 
                 sigma0=1.0,     
                 first_level=0, 
                 pyramid_type=PyramidType.RESIZE, 
                 use_block_adaptor=False,
                 do_parallel = kPyramidAdaptorUseParallelComputations,
                 do_sat_features_per_level = False):    
        self.detector = detector 
        self.descriptor = descriptor    
        self.num_features = num_features
        self.is_detector_equal_to_descriptor = (self.detector == self.descriptor)             
        self.num_levels = num_levels
        self.scale_factor = scale_factor 
        self.inv_scale_factor = 1./scale_factor         
        self.sigma0 = sigma0  
        self.first_level = first_level
        self.pyramid_type = pyramid_type
        self.use_block_adaptor = use_block_adaptor
        self.do_parallel = do_parallel   
        self.do_sat_features_per_level = do_sat_features_per_level  
                
        self.pyramid = Pyramid(num_levels=num_levels, 
                               scale_factor=scale_factor, 
                               sigma0=sigma0,
                               first_level=first_level,
                               pyramid_type=pyramid_type)
        self.initSigmaLevels()
        
        self.block_adaptor = None 
        if self.use_block_adaptor:
            self.block_adaptor = BlockAdaptor(self.detector, self.descriptor, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs, do_parallel=False)            


    def initSigmaLevels(self): 
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.scale_factors = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.scale_factors[0]=1.0          
        self.num_features_per_level = np.zeros(num_levels,dtype=np.int)
        num_desired_features_per_level = self.num_features*(1 - self.inv_scale_factor)/(1 - math.pow(self.inv_scale_factor, self.num_levels))
        sum_num_features = 0
        for level in range(self.num_levels-1):
            self.num_features_per_level[level] = int(round(num_desired_features_per_level))
            sum_num_features += self.num_features_per_level[level];
            num_desired_features_per_level *= self.inv_scale_factor
        self.num_features_per_level[self.num_levels-1] = max(self.num_features - sum_num_features, 0)    
        
        if self.first_level==-1:
            self.scale_factors[0]=1.0/self.scale_factor                   
        self.inv_scale_factors[0]=1.0/self.scale_factors[0]        
        for i in range(1,num_levels):
            self.scale_factors[i]=self.scale_factors[i-1]*self.scale_factor
            self.inv_scale_factors[i]=1.0/self.scale_factors[i]
                 
    def detect(self, frame, mask=None):      
        if self.num_levels == 1: 
            return self.detector.detect(frame, mask)
        else:    
            if kVerbose:              
                print('PyramidAdaptor #levels:', self.num_levels,'(from',self.first_level,'), scale_factor:', self.scale_factor,', sigma0:', self.sigma0,', type:', self.pyramid_type.name)
            self.pyramid.compute(frame)
            kps_all = []  
            
            def detect_level(scale,pyr_cur,i):
                kps = [] 
                if self.block_adaptor is None:        
                    kps = self.detector.detect(pyr_cur)                 
                else:
                    kps = self.block_adaptor.detect(pyr_cur)              
                for kp in kps:
                    kp.pt = (kp.pt[0]*scale, kp.pt[1]*scale) 
                    kp.size = kp.size*scale   
                    kp.octave = i      
                if self.do_sat_features_per_level:     
                    kps, _ = sat_num_features(kps, None, self.num_features_per_level[i])  # experimental                                                                                    
                kps_all.extend(kps)
                
            if not self.do_parallel:
                for i in range(0,self.num_levels):              
                    scale = self.scale_factors[i]
                    pyr_cur  = self.pyramid.imgs[i]   
                    detect_level(scale,pyr_cur,i)  
            else:          
                futures = []
                with ThreadPoolExecutor(max_workers = 4) as executor:
                    for i in range(0,self.num_levels):              
                        scale = self.scale_factors[i]
                        pyr_cur  = self.pyramid.imgs[i]                       
                        futures.append(executor.submit(detect_level, scale, pyr_cur, i))
                    wait(futures)                
                                    
            return np.array(kps_all)  
            
    def detectAndCompute(self, frame, mask=None):      
        if self.num_levels == 1: 
            return self.detector.detectAndCompute(frame, mask)
        else:    
            if kVerbose:              
                print('PyramidAdaptor [dc] #levels:', self.num_levels,'(from',self.first_level,'), scale_factor:', self.scale_factor,', sigma0:', self.sigma0,', type:', self.pyramid_type.name)
            self.pyramid.compute(frame)
            kps_all = []
            des_all = []            
            kps_des_map = {} 
                        
            def detect_and_compute_level(scale, pyr_cur, pyr_cur_filtered, N, i):                                       
                kps = [] 
                if self.block_adaptor is None:        
                    if self.is_detector_equal_to_descriptor:             
                        kps, des = self.detector.detectAndCompute(pyr_cur)
                    else:
                        kps = self.detector.detect(pyr_cur)                          
                        kps, des = self.descriptor.compute(pyr_cur_filtered, kps)                                            
                else:
                    kps, des = self.block_adaptor.detectAndCompute(pyr_cur)               
                for kp in kps:
                    kp.pt = (kp.pt[0]*scale, kp.pt[1]*scale) 
                    kp.size = kp.size*scale   
                    kp.octave = i      
                if self.do_sat_features_per_level:                        
                     kps, des = sat_num_features(kps, des, N)  # experimental                  
                kps_des_map[i] = (kps,des)    
                                                                                      
            if not self.do_parallel:
                for i in range(0,self.num_levels):              
                    scale = self.scale_factors[i]
                    pyr_cur  = self.pyramid.imgs[i]    
                    pyr_cur_filtered  = self.pyramid.imgs_filtered[i]  
                    detect_and_compute_level(scale, pyr_cur, pyr_cur_filtered, self.num_features_per_level[i], i)
            else:          
                futures = []
                with ThreadPoolExecutor(max_workers = 4) as executor:
                    for i in range(0,self.num_levels):              
                        scale = self.scale_factors[i]
                        pyr_cur  = self.pyramid.imgs[i]    
                        pyr_cur_filtered  = self.pyramid.imgs_filtered[i]            
                        futures.append(executor.submit(detect_and_compute_level, scale, pyr_cur, pyr_cur_filtered, self.num_features_per_level[i], i))
                    wait(futures) 

            for i,(kps,des) in kps_des_map.items():
                kps_all.extend(kps)       
                if des is not None and len(des)>0:                    
                    if len(des_all)>0:
                        des_all = np.vstack([des_all, des])                                       
                    else:                  
                        des_all = des                                                                                
            return np.array(kps_all), np.array(des_all)          