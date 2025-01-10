import nibabel as nib 
import numpy as np 

def read_nii(filename):
    '''
        https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
        This function reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    image = nib.load(filename).get_fdata()
    return image

