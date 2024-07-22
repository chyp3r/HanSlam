from .feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from .parameters import Parameters  

kNumFeatures=Parameters.kNumFeatures    

class FeatureManagerConfigs(object):   
    TEMPLATE = dict(num_features=kNumFeatures,                     
                num_levels=8,     
                scale_factor = 1.2,                                     
                detector_type = FeatureDetectorTypes.NONE,
                descriptor_type = FeatureDescriptorTypes.NONE)      
            
    @staticmethod        
    def extract_from(dict_in):
        dict_out = { key:dict_in[key] for key in FeatureManagerConfigs.TEMPLATE.keys() if key in dict_in }      
        return dict_out       