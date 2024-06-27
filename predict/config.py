import configparser 
import os
import yaml
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class Config(object):
    def __init__(self):
        self.root_folder = __location__
        self.config_file = 'config.ini'
        self.config_parser = configparser.ConfigParser()
        self.cam_settings = None
        self.dataset_settings = None
        self.dataset_type = None
        self.config_parser.read(__location__ + '/' + self.config_file)
        self.get_dataset_settings()
        self.get_cam_settings()
            
    def get_dataset_settings(self):
        self.dataset_type = self.config_parser['DATASET']['type']
        self.dataset_settings = self.config_parser[self.dataset_type]

        self.dataset_path = self.dataset_settings['base_path'];
        self.dataset_settings['base_path'] = os.path.join( __location__, self.dataset_path)
        print('dataset_settings: ', self.dataset_settings)

    def get_cam_settings(self):
        self.cam_settings = None
        self.settings_doc = __location__ + '/' + self.config_parser[self.dataset_type]['cam_settings']
        print(self.settings_doc)
        if(self.settings_doc is not None):
            with open(self.settings_doc, 'r') as stream:
                try:
                    self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)

    @property
    def K(self):
        if not hasattr(self, '_K'):
            fx = self.cam_settings['Camera.fx']
            cx = self.cam_settings['Camera.cx']
            fy = self.cam_settings['Camera.fy']
            cy = self.cam_settings['Camera.cy']
            self._K = np.array([[fx,  0, cx],
                                [ 0, fy, cy],
                                [ 0,  0,  1]])
        return self._K

    @property
    def Kinv(self):
        if not hasattr(self, '_Kinv'):
            fx = self.cam_settings['Camera.fx']
            cx = self.cam_settings['Camera.cx']
            fy = self.cam_settings['Camera.fy']
            cy = self.cam_settings['Camera.cy']
            self._Kinv = np.array([[1/fx,    0, -cx/fx],
                                   [   0, 1/fy, -cy/fy],
                                   [   0,    0,      1]])
        return self._Kinv

    @property
    def DistCoef(self):
        if not hasattr(self, '_DistCoef'):
            k1 = self.cam_settings['Camera.k1']
            k2 = self.cam_settings['Camera.k2']
            p1 = self.cam_settings['Camera.p1']
            p2 = self.cam_settings['Camera.p2']
            k3 = 0
            if 'Camera.k3' in self.cam_settings:
                k3 = self.cam_settings['Camera.k3']
            self._DistCoef = np.array([k1, k2, p1, p2, k3])
        return self._DistCoef

    @property
    def width(self):
        if not hasattr(self, '_width'):
            self._width = self.cam_settings['Camera.width']
        return self._width

    @property
    def height(self):
        if not hasattr(self, '_height'):
            self._height = self.cam_settings['Camera.height']
        return self._height
    
    @property
    def fps(self):
        if not hasattr(self, '_fps'):
            self._fps= self.cam_settings['Camera.fps']
        return self._fps    


if __name__ != "__main__":
    cfg = Config()
