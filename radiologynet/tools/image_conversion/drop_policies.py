import string
from numpy import array
import numpy as np

def shape_policy (pixel_data: array, threshold: float) -> array:
    """
    Function that test if pixel data is present and if the shape policy is in a given threshold

    Args:
        * pixel_data, ndarray -> Array of pixel data. Obtainable by get_images() method from dicom.py
        * threshold, float -> threshold representing ratio below images are dropped


    Returns:
        * array, ndarray -> Array contating all valid pixel data with shape ratio above threshold, or
        False if array is empty after applying shape_policy
    
    """

    #Check for single image
    if min(pixel_data.shape) / max(pixel_data.shape) < threshold:
        return False
    else:
        return True
    
    
def value_policy (pixel_data: array, threshold: float) -> array:
    """
    Function that test if pixel data is having enough diversity

    Args:
        * dcm_path: Path to dcm file 
        * threshold, float -> threshold representing percentage oh how many different values image
        must contain


    Returns:
        * bool, -> True if ratio of data fulfillness is above set threhsold.
    
    """
    _bits = np.max(np.max(pixel_data))
    # Black image
    if _bits < 1:
        return False
    _bits = int(np.log2(_bits)) + 1  
    
    # Near black image
    if _bits < 7:
        return False
    _data = np.ravel(pixel_data)
    _hist, _bins = np.histogram(_data, bins = np.linspace(0, pow(2, _bits)-1, pow(2, _bits)), density = False)
    
    _ratio = np.count_nonzero(_hist) / (pow(2, _bits) - 1) 
    if _ratio > threshold:
        return True
    else:
        return False

   