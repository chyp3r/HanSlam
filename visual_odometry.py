import numpy as np 
import cv2
from enum import Enum

from feature_tracker import FeatureTrackerTypes, FeatureTracker
from utils_geom import poseRt
from timer import TimerFps

class VoStage(Enum):
    NO_IMAGES_YET   = 0    
    GOT_FIRST_IMAGE = 1     
    
kVerbose=True     
kMinNumFeature = 2000
kRansacThresholdNormalized = 0.0003  
kRansacThresholdPixels = 0.1         
kAbsoluteScaleThreshold = 0.1        
kUseEssentialMatrixEstimation = True 
kRansacProb = 0.999
kUseGroundTruthScale = True 

class VisualOdometry(object):
    def __init__(self, cam, groundtruth, feature_tracker : FeatureTracker):
        self.stage = VoStage.NO_IMAGES_YET
        self.cam = cam
        self.cur_image = None  
        self.prev_image = None 

        self.kps_ref = None  
        self.des_ref = None 
        self.kps_cur = None 
        self.des_cur = None 

        self.cur_R = np.eye(3,3) 
        self.cur_t = np.zeros((3,1)) 

        self.trueX, self.trueY, self.trueZ = None, None, None
        self.groundtruth = groundtruth
        
        self.feature_tracker = feature_tracker
        self.track_result = None 

        self.mask_match = None 
        self.draw_img = None 

        self.init_history = True 
        self.poses = []              
        self.t0_est = None            
        self.t0_gt = None            
        self.traj3d_est = []        
        self.traj3d_gt = []           

        self.num_matched_kps = None    
        self.num_inliers = None       

        self.timer_verbose = False 
        self.timer_main = TimerFps('VO', is_verbose = self.timer_verbose)
        self.timer_pose_est = TimerFps('PoseEst', is_verbose = self.timer_verbose)
        self.timer_feat = TimerFps('Feature', is_verbose = self.timer_verbose)

    def getAbsoluteScale(self, frame_id):  
        if self.groundtruth is not None and kUseGroundTruthScale:
            self.trueX, self.trueY, self.trueZ, scale = self.groundtruth.getPoseAndAbsoluteScale(frame_id)
            return scale
        else:
            self.trueX = 0 
            self.trueY = 0 
            self.trueZ = 0
            return 1

    def computeFundamentalMatrix(self, kps_ref, kps_cur):
            F, mask = cv2.findFundamentalMat(kps_ref, kps_cur, cv2.FM_RANSAC, param1=kRansacThresholdPixels, param2=kRansacProb)
            if F is None or F.shape == (1, 1):
                raise Exception('No fundamental matrix found')
            elif F.shape[0] > 3:
                F = F[0:3, 0:3]
            return np.matrix(F), mask 	

    def removeOutliersByMask(self, mask): 
        if mask is not None:    
            n = self.kpn_cur.shape[0]     
            mask_index = [ i for i,v in enumerate(mask) if v > 0]    
            self.kpn_cur = self.kpn_cur[mask_index]           
            self.kpn_ref = self.kpn_ref[mask_index]           
            if self.des_cur is not None: 
                self.des_cur = self.des_cur[mask_index]        
            if self.des_ref is not None: 
                self.des_ref = self.des_ref[mask_index]  
            if kVerbose:
                print('removed ', n-self.kpn_cur.shape[0],' outliers')                

    def estimatePose(self, kps_ref, kps_cur):	
        kp_ref_u = self.cam.undistort_points(kps_ref)	
        kp_cur_u = self.cam.undistort_points(kps_cur)	        
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)
        if kUseEssentialMatrixEstimation:
            ransac_method = None 
            try: 
                ransac_method = cv2.USAC_MSAC 
            except: 
                ransac_method = cv2.RANSAC
            E, self.mask_match = cv2.findEssentialMat(self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.), method=ransac_method, prob=kRansacProb, threshold=kRansacThresholdNormalized)
        else:
            F, self.mask_match = self.computeFundamentalMatrix(kp_cur_u, kp_ref_u)
            E = self.cam.K.T @ F @ self.cam.K   
        _, R, t, mask = cv2.recoverPose(E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0., 0.))   
        return R,t  
    def processFirstFrame(self):
        self.kps_ref, self.des_ref = self.feature_tracker.detectAndCompute(self.cur_image)
        self.kps_ref = np.array([x.pt for x in self.kps_ref], dtype=np.float32) 
        self.draw_img = self.drawFeatureTracks(self.cur_image)

    def processFrame(self, frame_id):
        self.timer_feat.start()
        self.track_result = self.feature_tracker.track(self.prev_image, self.cur_image, self.kps_ref, self.des_ref)
        self.timer_feat.refresh()
        self.timer_pose_est.start()
        R, t = self.estimatePose(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)     
        self.timer_pose_est.refresh()
        self.kps_ref = self.track_result.kps_ref
        self.kps_cur = self.track_result.kps_cur
        self.des_cur = self.track_result.des_cur 
        self.num_matched_kps = self.kpn_ref.shape[0] 
        self.num_inliers =  np.sum(self.mask_match)
        # if kVerbose:        
        #     print('# matched points: ', self.num_matched_kps, ', # inliers: ', self.num_inliers)      
        absolute_scale = self.getAbsoluteScale(frame_id)
        if(absolute_scale > kAbsoluteScaleThreshold):
            # print('estimated t with norm |t|: ', np.linalg.norm(t), ' (just for sake of clarity)')
            self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
            self.cur_R = self.cur_R.dot(R)       
        self.draw_img = self.drawFeatureTracks(self.cur_image) 
        if (self.feature_tracker.tracker_type == FeatureTrackerTypes.LK) and (self.kps_ref.shape[0] < self.feature_tracker.num_features): 
            self.kps_cur, self.des_cur = self.feature_tracker.detectAndCompute(self.cur_image)           
            self.kps_cur = np.array([x.pt for x in self.kps_cur], dtype=np.float32) 
            # if kVerbose:     
            #     print('# new detected points: ', self.kps_cur.shape[0])                  
        self.kps_ref = self.kps_cur
        self.des_ref = self.des_cur
        self.updateHistory()           
        

    def track(self, img, frame_id):
        if kVerbose:
            print('..................................')
            print('frame: ', frame_id) 
        if img.ndim>2:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)             
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.cur_image = img
        if(self.stage == VoStage.GOT_FIRST_IMAGE):
            self.processFrame(frame_id)
        elif(self.stage == VoStage.NO_IMAGES_YET):
            self.processFirstFrame()
            self.stage = VoStage.GOT_FIRST_IMAGE            
        self.prev_image = self.cur_image    
        self.timer_main.refresh()  
  

    def drawFeatureTracks(self, img, reinit = False):
        draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        num_outliers = 0        
        if(self.stage == VoStage.GOT_FIRST_IMAGE):            
            if reinit:
                for p1 in self.kps_cur:
                    a,b = p1.ravel()
                    cv2.circle(draw_img,(a,b),1, (0,255,0),-1)                    
            else:    
                for i,pts in enumerate(zip(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)):
                    drawAll = False 
                    if self.mask_match[i] or drawAll:
                        p1, p2 = pts 
                        a,b = p1.astype(int).ravel()
                        c,d = p2.astype(int).ravel()
                        cv2.line(draw_img, (a,b),(c,d), (0,255,0), 1)
                        cv2.circle(draw_img,(a,b),1, (0,0,255),-1)   
                    else:
                        num_outliers+=1
            # if kVerbose:
            #     print('# outliers: ', num_outliers)     
        return draw_img            

    def updateHistory(self):
        if (self.init_history is True) and (self.trueX is not None):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  
            self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])           
            self.init_history = False 
        if (self.t0_est is not None) and (self.t0_gt is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   
            self.traj3d_est.append(p)
            pg = [self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]]
            self.traj3d_gt.append(pg)     
            self.poses.append(poseRt(self.cur_R, p))   
