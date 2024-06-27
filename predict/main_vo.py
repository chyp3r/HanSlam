import numpy as np
import time

from .config import Config
from .visual_odometry import VisualOdometry
from .camera  import PinholeCamera
from .dataset import dataset_factory

from .feature_tracker import feature_tracker_factory 

from .feature_tracker_configs import FeatureTrackerConfigs

import cv2


class TrackerModel:
    
    def __init__(self, translation) -> None:
        
        self.TRANSLATION_X = translation["translation_x"]
        self.TRANSLATION_Y = translation["translation_y"]
        
        config = Config()   

        #self.dataset = dataset_factory(config.dataset_settings)

        cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                            config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                            config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                            config.DistCoef, config.cam_settings['Camera.fps'])

        num_features=1000 
        tracker_config = FeatureTrackerConfigs.BRISK
        tracker_config['num_features'] = num_features
        feature_tracker1 = feature_tracker_factory(**tracker_config)


        self.vo = VisualOdometry(cam, None, feature_tracker1)
        self.img_id_counter = 0
        self.config = config
        
        """
        traj_img_size = 800
        traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
        half_traj_img_size = int(0.5*traj_img_size)
        draw_scale = 1
        img_id = 0
        """
        

    def path_tracker(self, image_path):
        baslangic_zamani = time.time()
        img = self.getImage(image_path)
        x,y,z = None, None, None
        if img is not None:
            self.vo.track(img,self.img_id_counter)
            if(self.img_id_counter > 2):	       
                x, y, z = self.vo.traj3d_est[-1]
                #coor = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x,y,z)
                #print(coor)

            bitis_zamani = time.time()
            #print(f"fps {1/(bitis_zamani-baslangic_zamani)}")
            self.img_id_counter += 1
        if x is None:
            return {"x": x, "y": y, "z": z}
        else:
            return {"x": x[0] + self.TRANSLATION_X, "y": y[0]+ self.TRANSLATION_Y, "z": z[0]}
                
    def getImage(self, image_path):
        """
        if self.i == self.maxlen:
            return None
        image_file = self.listing[self.i]
        img = cv2.imread(image_file)
        """
        img = cv2.imread(image_path)     
        if img is None: 
            raise IOError('error reading file: ', image_path)               

        if img.ndim == 3:
            grayValue = 0.07 * img[:,:,2] + 0.72 * img[:,:,1] + 0.21 * img[:,:,0]
            gray_img = grayValue.astype(np.uint8)
            return gray_img   
        else:
            return img 



                