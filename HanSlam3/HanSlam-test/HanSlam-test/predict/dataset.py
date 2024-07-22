import numpy as np 
import cv2
import os
import glob
from .utils_sys import Printer 
from enum import Enum

class DatasetType(Enum):
    NONE = 1
    KITTI = 2
    TUM = 3
    VIDEO = 4
    FOLDER = 5  

def dataset_factory(settings,i = 0):
    type=DatasetType.NONE
    associations = None    
    path = None 

    type = settings['type']
    name = settings['name']    
    path = settings['base_path'] 
    path = os.path.expanduser(path)
    
    if 'associations' in settings:
        associations = settings['associations']
    dataset = None 
    if type == 'folder':
        fps = 60 
        if 'fps' in settings:
            fps = int(settings['fps'])
        dataset = FolderDataset(path, name, fps, associations, DatasetType.FOLDER,i) 
       
    return dataset 

class Dataset(object):
    def __init__(self, path, name, fps=None, associations=None, type=DatasetType.NONE):
        self.path=path 
        self.name=name 
        self.type=type    
        self.is_ok = True
        self.fps = fps   
        if fps is not None:       
            self.Ts = 1./fps 
        else: 
            self.Ts = None 
          
        self.timestamps = None 
        self._timestamp = None      
        self._next_timestamp = None  
        
    def isOk(self):
        return self.is_ok

    def getImage(self, frame_id):
        return None 

    def getImage1(self, frame_id):
        return None

    def getDepth(self, frame_id):
        return None        

    def getImageColor(self, frame_id):
        try: 
            img = self.getImage(frame_id)
            if img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None  
            Printer.red('Cannot open dataset: ', self.name, ', path: ', self.path)
            return img    
        
    def getTimestamp(self):
        return self._timestamp
    
    def getNextTimestamp(self):
        return self._next_timestamp    

class FolderDataset(Dataset): 
    def __init__(self, path, name, fps=None, associations=None, type=DatasetType.VIDEO,i = 0): 
        super().__init__(path, name, fps, associations, type)  
        if fps is None: 
            fps = 20   
        self.Ts = 1./self.fps 
        self.listing = []    
        self.maxlen = 1000000    
        self.listing = glob.glob(path + '\\' + self.name)
        self.listing.sort()
        self.listing = self.listing[i:]
        self.maxlen = len(self.listing)
        self.i = 0      
        self._timestamp = 0.        
            
    def getImage(self, imagePath):
        """
        if self.i == self.maxlen:
            return None
        image_file = self.listing[self.i]
        img = cv2.imread(image_file)
        """
        img = cv2.imread(imagePath)
        self._timestamp += self.Ts
        self._next_timestamp = self._timestamp + self.Ts         
        if img is None: 
            raise IOError('error reading file: ', imagePath)               
        self.i = self.i + 1
        if img.ndim == 3:
            grayValue = 0.07 * img[:,:,2] + 0.72 * img[:,:,1] + 0.21 * img[:,:,0]
            gray_img = grayValue.astype(np.uint8)
            return gray_img   
        else:
            return img 