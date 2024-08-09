# IMPORTS
import numpy as np
from skimage.morphology import erosion, dilation, ball

def morphOps(mask,region='whole lesion',r=2):
    """
        Apply morphological operations to a binary mask based on the specified region.
        Parameters:
        - mask (ndarray): Binary mask representing the region of interest (assumes that mask and image have been resampled to [1,1,1] pixel spacing).
        - region (str): Region of interest to apply morphological operations. Default is 'whole lesion'.
                        Possible values: 'whole lesion', 'lesion core', 'interior rim', 'exterior rim', 'peripheral ring'.
        - r (int): Radius parameter for morphological operations. Default is 2.
        Returns:
        - morphMsk (ndarray): Binary mask after applying the specified morphological operations.
    """
    
    mask = mask == 1 # force-convert to binary mask

    morphFunctions = { 'whole lesion' :  lambda x: 1 * x,
                       'lesion core' :  lambda x: 1 * erosion(x,ball(radius = r)),
                       'interior rim' :  lambda x: 1 * np.logical_and(x,~erosion(x,ball(radius = r))),
                       'exterior rim' :  lambda x: 1 * np.logical_and(~x,dilation(x,ball(radius = r))),
                       'peripheral ring' :  lambda x: 1 * np.logical_and(~erosion(x,ball(radius = r)),dilation(x,ball(radius = r))) 
                       }     
        
    return morphFunctions[region](mask)  