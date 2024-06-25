import numpy as np
import time

from config import Config
from visual_odometry import VisualOdometry
from camera  import PinholeCamera
from dataset import dataset_factory

from feature_tracker import feature_tracker_factory 

from feature_tracker_configs import FeatureTrackerConfigs

if __name__ == "__main__":

    config = Config()   

    dataset = dataset_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])

    num_features=1000 
    tracker_config = FeatureTrackerConfigs.ORB
    tracker_config['num_features'] = num_features
    feature_tracker1 = feature_tracker_factory(**tracker_config)


    vo = VisualOdometry(cam, None, feature_tracker1)

    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1
    img_id = 0

    while dataset.isOk():
        baslangic_zamani = time.time()
        img = dataset.getImage(img_id)
        if img is not None:
            vo.track(img,img_id)
            if(img_id > 2):	       
                x, y, z = vo.traj3d_est[-1]
                coor = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x,y,z)
                print(coor)

            bitis_zamani = time.time()
            print(f"fps {1/(bitis_zamani-baslangic_zamani)}")
            img_id += 1
        else:
            dataset = dataset_factory(config.dataset_settings,i=img_id)


                