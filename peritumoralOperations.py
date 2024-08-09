# IMPORTS
import numpy as np
from skimage.morphology import erosion, dilation, ball

def morphOps(mask,region='whole lesion',r=2):
    """
        Apply morphological operations to a binary mask based on the specified region.
        Parameters:
        - mask (ndarray): Binary mask representing the region of interest.
        - region (str): Region of interest to apply morphological operations. Default is 'whole lesion'.
                        Possible values: 'whole lesion', 'lesion core', 'interior rim', 'exterior rim', 'peripheral ring'.
        - r (int): Radius parameter for morphological operations. Default is 2.
        Returns:
        - morphMsk (ndarray): Binary mask after applying the specified morphological operations.
    """
    
    mask = mask == 1 # force-convert to binary mask
            
    if region == 'whole lesion':
        morphMsk = 1 * mask
    
    if region == 'lesion core':
        morphMsk = 1 * erosion(mask,ball(radius = r))

    if region == 'interior rim':
        morphMsk = 1 * np.logical_and(mask,~erosion(mask,ball(radius = r)))
        
    if region == 'exterior rim':
        morphMsk = 1 * np.logical_and(~mask,dilation(mask,ball(radius = r)))
        
    if region == 'peripheral ring':
        morphMsk = 1 * np.logical_and(~erosion(mask,ball(radius = r)),dilation(mask,ball(radius = r)))   
        
    return morphMsk