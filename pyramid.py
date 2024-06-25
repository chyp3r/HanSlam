import math
from enum import Enum
 
import numpy as np 
import cv2

from utils_img import img_blocks
from utils_sys import Printer 


kVerbose = True   

kNumLevelsInitSigma = 20

class PyramidType(Enum):
    RESIZE            = 0 
    RESIZE_AND_FILTER = 1  
    GAUSS_PYRAMID     = 2  

class Pyramid(object): 
    def __init__(self, num_levels=4, scale_factor=1.2, 
                 sigma0=1.0,     
                 first_level=0,  
                 pyramid_type=PyramidType.RESIZE):             
        self.num_levels = num_levels
        self.scale_factor = scale_factor 
        self.sigma0 = sigma0  
        self.first_level = first_level
        self.pyramid_type = pyramid_type
                
        self.imgs       = [] 
        self.imgs_filtered = []     
        self.base_img = None      
        
        self.scale_factors = None
        self.inv_scale_factors = None     
        self.initSigmaLevels()          
        
        
    def initSigmaLevels(self): 
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.scale_factors = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.scale_factors[0]=1.0              
        self.inv_scale_factors[0]=1.0/self.scale_factors[0]        
        for i in range(1,num_levels):
            self.scale_factors[i]=self.scale_factors[i-1]*self.scale_factor
            self.inv_scale_factors[i]=1.0/self.scale_factors[i]   

    def compute(self, frame):
        if self.first_level == -1:
            frame = self.createBaseImg(frame)  
        if self.pyramid_type == PyramidType.RESIZE:
            return self.computeResize(frame)
        elif self.pyramid_type == PyramidType.RESIZE_AND_FILTER:          
            return self.computeResizeAndFilter(frame)              
        elif self.pyramid_type == PyramidType.GAUSS_PYRAMID:
            return self.computeGauss(frame) 
        else: 
            Printer.orange('Pyramid - unknown type')    
            return self.computeResizePyramid(frame)                    


    def createBaseImg(self, frame):
        sigma_init = 0.5  
        delta_sigma = math.sqrt( max(self.sigma0*self.sigma0 - (sigma_init*sigma_init*self.scale_factor*self.scale_factor), 0.01) )
        frame_upscaled = cv2.resize(frame,(0,0),fx=self.scale_factor,fy=self.scale_factor)
        if self.pyramid_type == PyramidType.GAUSS_PYRAMID:
            return cv2.GaussianBlur(frame_upscaled,ksize=(0,0),sigmaX=delta_sigma)
        else: 
            return frame_upscaled
        
        
    def computeResize(self, frame): 
        inv_scale = 1./self.scale_factor
        self.imgs = []        
        self.imgs_filtered = []  
        pyr_cur = frame 
        for i in range(0,self.num_levels):
            self.imgs.append(pyr_cur)          
            self.imgs_filtered.append(pyr_cur)                     
            if i < self.num_levels-1:    
                pyr_down = cv2.resize(pyr_cur,(0,0),fx=inv_scale,fy=inv_scale)                 
                pyr_cur  = pyr_down        
      
    def computeResizeAndFilter(self, frame): 
        inv_scale = 1./self.scale_factor
        filter_sigmaX = 2  
        ksize=(5,5)
        self.imgs = []        
        self.imgs_filtered = []  
        pyr_cur = frame 
        for i in range(0,self.num_levels):
            filtered = cv2.GaussianBlur(pyr_cur,ksize,sigmaX=filter_sigmaX) 
            self.imgs.append(pyr_cur)          
            self.imgs_filtered.append(filtered)                         
            if i < self.num_levels-1:    
                pyr_down = cv2.resize(pyr_cur,(0,0),fx=inv_scale,fy=inv_scale)               
                pyr_cur  = pyr_down            
          
                       
    def computeGauss(self, frame): 
        inv_scale = 1./self.scale_factor
        
       
        sigma_nominal = 0.5  
        sigma0 = self.sigma0  
        sigma_prev = sigma_nominal
                
        self.imgs = []        
        self.imgs_filtered = []  
                
        pyr_cur = frame 
                          
        for i in range(0,self.num_levels):
            if i == 0 and self.first_level == -1:
                sigma_prev = sigma0 
                filtered = frame               
            else:     
                sigma_total = self.scale_factors[i] * sigma0
                sigma_cur = math.sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev) 
                sigma_prev = sigma_cur  
                filtered = cv2.GaussianBlur(pyr_cur,ksize=(0,0),sigmaX=sigma_cur) 
                
            self.imgs.append(filtered)           
            self.imgs_filtered.append(filtered)                      
                            
            if i < self.num_levels-1:           
                pyr_down = cv2.resize(filtered,(0,0),fx=inv_scale,fy=inv_scale)    
                pyr_cur  = pyr_down    

