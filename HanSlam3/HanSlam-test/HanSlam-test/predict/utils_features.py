import math 
import numpy as np
import cv2

from enum import Enum
from scipy.spatial import cKDTree
from .utils_sys import Printer, import_from, is_opencv_version_greater_equal
from .utils_geom import add_ones, s1_diff_deg, s1_dist_deg, l2_distances
from .parameters import Parameters 

ORBextractor = import_from('orbslam2_features', 'ORBextractor')  
kPySlamUtilsAvailable = False 
Printer.orange('WARNING: cannot import pyslam_utils')
      

def convert_pts_to_keypoints(pts, size=1): 
    kps = []
    if pts is not None: 
        if pts.ndim > 2:
            if is_opencv_version_greater_equal(4,5,3):  
                kps = [ cv2.KeyPoint(p[0][0], p[0][1], size=size) for p in pts ]          
            else:
                kps = [ cv2.KeyPoint(p[0][0], p[0][1], _size=size) for p in pts ]
        else: 
            # convert matrix [Nx2] of pts into list of keypoints  
            if is_opencv_version_greater_equal(4,5,3):  
                kps = [ cv2.KeyPoint(p[0], p[1], size=size) for p in pts ]                      
            else:
                kps = [ cv2.KeyPoint(p[0], p[1], _size=size) for p in pts ]     
    return kps         

def unpackSiftOctave(kpt):
    _octave = kpt.octave
    octave = int(_octave)&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1.0/(1<<octave))
    else:
        scale = float(1<<(-octave))
    return (octave, layer, scale)
    
def unpackSiftOctavePlusOne(kpt):
    """unpackSiftOctavePlusOne(kpt)->octave+1 (-1 is upsampled layer, 0 is input image layer, 1 is the first layer and so on... )
    @brief Unpack Sift Keypoint
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    if octave>=128:
        octave |= -128        
    return octave+1    

# return a virtual 'intra'-level with a virtual scale factor =  2^(1/3) where 3 is the number of intra-layers per octave    
def unpackSiftOctaveIntra(kpt):
    """unpackSiftOctaveVirtual(kpt)-> (octave+1)*3 + layer
    @brief Unpack Sift Keypoint; return a virtual 'intra' level with a virtual scale factor =  2^(1/3) where 3 is the number of intra-layers per octave
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = int(_octave)&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1.0/(1<<octave))
    else:
        scale = float(1<<(-octave))
    return (octave+1)*3 + layer   # return a virtual intra-level with a virtual scale factor =  2^(1/3) where 3 is the number of layers per octave

class UnpackOctaveMethod(Enum):
    DEFAULT       = 1
    PLUS_ONE      = 2
    INTRAL_LAYERS = 3    

def unpackSiftOctaveKps(kps, method=UnpackOctaveMethod.DEFAULT):  
    if method == UnpackOctaveMethod.DEFAULT:
        for kpt in kps: 
            kpt.octave,_,_ = unpackSiftOctave(kpt)
    elif method == UnpackOctaveMethod.PLUS_ONE:
        for kpt in kps: 
            kpt.octave = unpackSiftOctavePlusOne(kpt)   
    elif method == UnpackOctaveMethod.INTRAL_LAYERS:           
        for kpt in kps: 
            kpt.octave = unpackSiftOctaveIntra(kpt)            
        
        
# robust estimatation of descriptor distance standard deviation by using MAD (Median Absolute Deviation)      
# N.B: you can use the thresholding condition: 
#      descriptor_distance < factor * sigma_mad
# https://en.wikipedia.org/wiki/Median_absolute_deviation 
def descriptor_sigma_mad(des1, des2, descriptor_distances=l2_distances):
    dists = np.full(des1.shape[0], 0., dtype=np.float32) 
    # for i in range(des1.shape[0]):
    #     dists[i] = descriptor_distance(des1[i],des2[i])
    dists = descriptor_distances(des1,des2)
    dists_median = np.median(dists)     # MAD, approximating dists_median=0 
    sigma_mad = 1.4826 * dists_median
    return sigma_mad, dists        

# robust estimation of descriptor distance standard deviation by using MAD (Median Absolute Deviation)  
# N.B: you can use the thresholding condition:
#  (descriptor_distance < dists_median) or (descriptor_distance - dists_median < factor * sigma_mad)
# https://en.wikipedia.org/wiki/Median_absolute_deviation
def descriptor_sigma_mad_v2(des1, des2, descriptor_distances=l2_distances):
    dists = np.full(des1.shape[0], 0., dtype=np.float32) 
    # for i in range(des1.shape[0]):
    #     dists[i] = descriptor_distance(des1[i],des2[i])
    dists = descriptor_distances(des1,des2)
    dists_median = np.median(dists)
    ads = np.fabs(dists - dists_median) # absolute deviations from median 
    sigma_mad = 1.4826 * np.median(ads) 
    return sigma_mad, dists_median, dists       


# keep the first 'self.num_features' best features
def sat_num_features(kps, des=None, num_features=Parameters.kNumFeatures):    
    if len(kps) > num_features:
        # keep the features with the best response 
        if des is None: 
            kps = sorted(kps, key=lambda x:x.response, reverse=True)[:num_features]     
        else:            
            # sort by score to keep highest score features 
            # print('sat with des')
            order = np.argsort([kp.response for kp in kps])[::-1][:num_features]      # [::-1] is for reverse order 
            kps = np.array(kps)[order]    
            des = np.array(des)[order]             
    return kps, des 

     
# kdtree-based non-maximum suppression of keypoints 
# adapted and optimized from https://stackoverflow.com/questions/9210431/well-distributed-features-using-opencv/50267891
def kdt_nms(kps, des=None, num_features=Parameters.kNumFeatures, r=Parameters.kKdtNmsRadius, k_max=9):
    """ Use kd-tree to perform local non-maximum suppression of key-points
    kps - key points obtained by one of openCVs 2d features detectors (SIFT, SURF, AKAZE etc..)
    r - the radius of points to query for removal
    k_max - maximum points retreived in single query (not used)
    """
    
    if len(kps)==0:
        return kps, des
    if des is not None:
        assert(len(des)==len(kps))
    
    # sort by score to give priority to highest score features 
    order = np.argsort([kp.response for kp in kps])[::-1]  # [::-1] is for reverse order     
    kps = np.array(kps)[order] 

    # create kd-tree for quick NN queries 
    data_pts = np.array([kp.pt for kp in kps],dtype=np.float32)
        
    kd_tree = cKDTree(data_pts)

    # perform NMS using kd-tree, by querying points by score order, 
    # and removing neighbors from future queries
    N = len(kps)
    idxs_removed = set() 
    
    #time_start = time.time()     
    kd_idxs = kd_tree.query_ball_point(data_pts,r) 
    #print('elapsed: ', time.time()-time_start)     
     
    for i in range(N):
        if i in idxs_removed:
            continue
        #idxs_removed.update([j for j in kd_idxs[i] if j >i])
        for j in kd_idxs[i]: 
            if j>i:
                idxs_removed.add(j)  
    idxs_remaining = [i for i in range(N) if i not in idxs_removed] 
    
    kps_out = kps[idxs_remaining]
    des_out = None
    if des is not None:
        #print('des.shape:',des.shape)
        des = des[order]    
        des_out = des[idxs_remaining]
    if len(kps_out) > num_features:
        kps_out = kps_out[:num_features]
        if des_out is not None:
            des_out = des_out[:num_features]                                
    return kps_out, des_out


# adapted from https://github.com/BAILOOL/ANMS-Codes
def ssc_nms(kps, des, cols, rows, num_ret_points=Parameters.kNumFeatures, tolerance=0.1):

    if len(kps)==0:
        return kps, des
        
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = 4 * cols + 4 * num_ret_points + 4 * rows * num_ret_points + rows * rows + cols * cols - \
           2 * rows * cols + 4 * rows * cols * num_ret_points
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = sol1 if (sol1 > sol2) else sol2  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(kps) / num_ret_points))

    prev_width = -1 
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if width == prev_width or low > high:  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [[False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_cols + 1)]
        result = []

        for i in range(len(kps)):
            row = int(math.floor(kps[i].pt[1] / c))  # get position of the cell current point is located at
            col = int(math.floor(kps[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range that current radius is covering
                row_min = int((row - math.floor(width / c)) if ((row - math.floor(width / c)) >= 0) else 0)
                row_max = int(
                    (row + math.floor(width / c)) if (
                            (row + math.floor(width / c)) <= num_cell_rows) else num_cell_rows)
                col_min = int((col - math.floor(width / c)) if ((col - math.floor(width / c)) >= 0) else 0)
                col_max = int(
                    (col + math.floor(width / c)) if (
                            (col + math.floor(width / c)) <= num_cell_cols) else num_cell_cols)
                for rowToCov in range(row_min, row_max + 1):
                    for colToCov in range(col_min, col_max + 1):
                        if not covered_vec[rowToCov][colToCov]:
                            covered_vec[rowToCov][colToCov] = True

        if k_min <= len(result) <= k_max:  
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  
        else:
            low = width + 1
        prev_width = width

    des_out = None
    if des is not None:
        des_out = des[result_list]
    kps_out = kps[result_list]                
    return kps_out, des_out

def octree_nms(frame, kps, num_features):
    minX=0
    maxX=frame.shape[1]
    minY=0
    maxY=frame.shape[0]
    kps_tuples = [ (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]
    kps_tuples = ORBextractor.DistributeOctTree(kps_tuples,minX,maxX,minY,maxY,num_features,0)
    kps = [ cv2.KeyPoint(*kp) for kp in kps_tuples]     
    return kps


def grid_nms(kps, des, H, W, num_features, dist_thresh=4):
    in_corners = np.array([(kp.pt[0],kp.pt[1],kp.response) for kp in kps]).T  #  3xN [x_i,y_i,conf_i]^T
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    #out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    kps_out = np.array(kps)[out_inds][:num_features]
    if des is not None: 
        des_out = des[out_inds][:num_features]
    else: 
        des_out = None 
    return kps_out, des_out, out_inds

def nms_from_map(score_map, size):
    kernel = np.ones((size,size),np.uint8)
    score_map = score_map * (score_map == cv2.dilate(score_map, kernel))  # cv2.dilate is a maximum filter 
    return score_map

def get_best_score_idxs(score_map, num_points=1000, threshold=-1):
    if threshold == -1:
        flatten = score_map.flatten()
        order_array = np.sort(flatten)[::-1]
        threshold = order_array[num_points-1]
        if threshold <= 0.0:
            indexes = np.argwhere(order_array > 0.0)
            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes)-1]]
    indexes = np.argwhere(score_map >= threshold)
    return indexes[:num_points]

def get_best_points_coordinates(score_map, num_points=1000, threshold=-1):
    indexes = get_best_score_idxs(score_map, num_points=num_points, threshold=threshold)
    coords = []
    for idx in indexes:
        score = score_map[idx[0], idx[1]]
        tmp = [idx[1], idx[0], score]
        coords.append(tmp)
    return np.asarray(coords)

def compute_hom_reprojection_error(H, kps1, kps2, mask=None):
    if mask is not None: 
        mask_idxs = (mask.ravel() == 1) 
        kps1 = kps1[mask_idxs]
        kps2 = kps2[mask_idxs]  
    kps1_reproj = H @ add_ones(kps1).T
    kps1_reproj = kps1_reproj[:2]/kps1_reproj[2]
    error_vecs = kps1_reproj.T - kps2
    return np.mean(np.sum(error_vecs*error_vecs,axis=1))
    
def extract_patches_tensor(img, kps, patch_size=32, mag_factor=1.0, warp_flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS):
    patches = np.ndarray((len(kps), 1, patch_size, patch_size), dtype=np.float32)
    half_patch_size=0.5*patch_size
    for i,kp in enumerate(kps):
        x,y = kp.pt
        s = kp.size
        a = kp.angle

        scale = mag_factor * s/patch_size
        a_rad = a * math.pi/180.0
        cos = math.cos(a_rad) if a_rad >=0 else 1.0 
        sin = math.sin(a_rad) if a_rad >=0 else 0.0 
        scale_cos = scale*cos
        scale_sin = scale*sin 

        M = np.matrix([
            [+scale_cos, -scale_sin, (-scale_cos + scale_sin) * half_patch_size + x],
            [+scale_sin, +scale_cos, (-scale_sin - scale_cos) * half_patch_size + y]])

        patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=warp_flags)
        patches[i,0,:,:] = cv2.resize(patch,(patch_size,patch_size))
    return patches  
            
            
def extract_patches_array(img, kps, patch_size=32, mag_factor=1.0, warp_flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS):
    patches = []
    half_patch_size=0.5*patch_size
    for kp in kps:
        x,y = kp.pt
        s = kp.size
        a = kp.angle

        scale = mag_factor * s/patch_size
        a_rad = a * math.pi/180.0
        cos = math.cos(a_rad) if a_rad >=0 else 1.0 
        sin = math.sin(a_rad) if a_rad >=0 else 0.0 
        scale_cos = scale*cos
        scale_sin = scale*sin 

        M = np.matrix([
            [+scale_cos, -scale_sin, (-scale_cos + scale_sin) * half_patch_size + x],
            [+scale_sin, +scale_cos, (-scale_sin - scale_cos) * half_patch_size + y]])

        patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=warp_flags)
        patches.append(patch)
    return patches 

def extract_patches_array_cpp(img, kps, patch_size=32, mag_factor=1.0, warp_flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS):
    print('using python version extract_patches_array()')
    return extract_patches_array(img=img, kps=kps, patch_size=patch_size, mag_factor=mag_factor, warp_flags=warp_flags)
