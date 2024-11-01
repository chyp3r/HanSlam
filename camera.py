import numpy as np 
import cv2
from utils_geom import add_ones

class Camera: 
    def __init__(self, width, height, fx, fy, cx, cy, D, fps = 1):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.D = np.array(D,dtype=np.float32)
        self.fps = fps 
        
        self.is_distorted = np.linalg.norm(self.D) > 1e-10
        self.initialized = False     
              
class PinholeCamera(Camera):
    def __init__(self, width, height, fx, fy, cx, cy, D, fps = 1):
        super().__init__(width, height, fx, fy, cx, cy, D, fps)
        self.K = np.array([[fx, 0,cx],
                           [ 0,fy,cy],
                           [ 0, 0, 1]])
        self.Kinv = np.array([[1/fx,    0,-cx/fx],
                              [   0, 1/fy,-cy/fy],
                              [   0,    0,    1]])             
        
        self.u_min, self.u_max = 0, self.width 
        self.v_min, self.v_max = 0, self.height       
        self.init()    
        
    def init(self):
        if not self.initialized:
            self.initialized = True 
            self.undistort_image_bounds()        

    def project(self, xcs):
        projs = self.K @ xcs.T     
        zs = projs[-1]      
        projs = projs[:2]/ zs   
        return projs.T, zs
        
    def unproject(self, uv):
        x = (uv[0] - self.cx)/self.fx
        y = (uv[1] - self.cy)/self.fy
        return x,y
  
    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]        


    def undistort_points(self, uvs):
        if self.is_distorted:
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)            
            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return uvs 
        
    def undistort_image_bounds(self):
        uv_bounds = np.array([[self.u_min, self.v_min],
                                [self.u_min, self.v_max],
                                [self.u_max, self.v_min],
                                [self.u_max, self.v_max]], dtype=np.float32).reshape(4,2)
        if self.is_distorted:
                uv_bounds_undistorted = cv2.undistortPoints(np.expand_dims(uv_bounds, axis=1), self.K, self.D, None, self.K)      
                uv_bounds_undistorted = uv_bounds_undistorted.ravel().reshape(uv_bounds_undistorted.shape[0], 2)
        else:
            uv_bounds_undistorted = uv_bounds 
        self.u_min = min(uv_bounds_undistorted[0][0],uv_bounds_undistorted[1][0])
        self.u_max = max(uv_bounds_undistorted[2][0],uv_bounds_undistorted[3][0])        
        self.v_min = min(uv_bounds_undistorted[0][1],uv_bounds_undistorted[2][1])    
        self.v_max = max(uv_bounds_undistorted[1][1],uv_bounds_undistorted[3][1])  

    def is_in_image(self, uv, z):
        return (uv[0] > self.u_min) & (uv[0] < self.u_max) & \
               (uv[1] > self.v_min) & (uv[1] < self.v_max) & \
               (z > 0)         
                            
    def are_in_image(self, uvs, zs):
        return (uvs[:, 0] > self.u_min) & (uvs[:, 0] < self.u_max) & \
               (uvs[:, 1] > self.v_min) & (uvs[:, 1] < self.v_max) & \
               (zs > 0 )
