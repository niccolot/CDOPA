import numpy as np
import pydicom
import os
from tensorflow import keras
import tensorflow_addons as tfa
import uuid # for generating guid code 

threshold = 7000 #intensity of pixel to ignore

#number of depth images to keep, only a certain number contains the base gangli
zmin = 20
zmax = 30

#portion of image which contains the interested part of brain scan
xmin = 33
xmax = 162
ymin = 73
ymax = 202

deltaX = xmax-xmin
deltaY = ymax-ymin
deltaZ = zmax-zmin

irangemin = zmin
irangemax = zmax

img_size = deltaX+1

background_percentage = 0.4

"""
DATA LOADING FUNCTIONS
"""

def load_scan(path):
    slices = [pydicom.read_file(os.path.join(path, s), force=True) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        
    return slices


def get_pixels(slices, rescale=False):
    if rescale:
        image = np.stack([s.pixel_array*(s.RescaleSlope*10) for s in slices])
        return np.array(image,dtype=np.float32)
    else:
        image = np.stack([s.pixel_array for s in slices])
        return np.array(image,dtype=np.int16)
    

def slicecutoff(slice, threshold):
    slice[slice<threshold]=0.
    return slice

    
def create_dataset(data_folder, rescale=False, threshold=threshold):

    patients = os.listdir(data_folder) #list all the folders containing the .dcm for each patient
    patients.sort()
    inumpatients = len(patients) #number of datapoints
    X_DATA = np.empty([inumpatients, img_size, img_size, deltaZ]) #initially 'empty dataset' to be filled
    
    for num_patient, patient in enumerate(patients):
        patient_n = get_pixels(load_scan(os.path.join(data_folder, patient)))

        #reference central slice and mask
        slice_central = patient_n[zmin+4]
        slice_centralcut = slice_central[xmin:xmax+1, ymin:ymax+1]

        if rescale:
            threshold_roi = slice_centralcut.max()*background_percentage
            threshold = threshold_roi

        mask_central = slicecutoff(slice_centralcut, threshold)

        for index, slice in enumerate(patient_n):
            #only central slices are considered for the dataset
            if index in range(zmin,zmax): 
                
                slicecut = slice[xmin:xmax+1, ymin:ymax+1]
                mask = slicecutoff(slicecut, threshold)

                #here it makes sure the larger portion of the mask is considered background
                #zeroing out the parts outsite the central region of interest
                if index < zmin + 4:
                    if (mask_central-mask).sum() > mask_central.sum():
                        slicecut[True] = 0.
                        mask[True] = 0.
                if index > zmin +4:
                    if abs((mask-mask_central)).sum() > mask_central.sum():
                        slicecut[True] = 0.
                        mask[True] = 0.
                
                X_DATA[num_patient,:,:,index-zmin] = slicecut
    
    return X_DATA


"""
DATASET MANIPULATION FUNCTIONS
"""

def shift_tensor(image, shift_param, axis='x'):
    """
    shift param is how many pixel the image has to be shifted, 
    it has to be a signed flot for tanslation to the left or right
    """

    if axis=='x':
        return tfa.image.translate(image, [shift_param,0])

    if axis=='y':
        return tfa.image.translate(image, [0, shift_param])

    else:
        raise Exception("Only x or y are valid axes")


def get_shifted_dataset(X_DATA, Y_D, Y_DATA, shifting_percentages):
    """
    returns the new dataset with the previous images and the translated ones
    """

    axes = ['x', 'y']

    augmented_dataset = X_DATA
    augmented_Y_D = Y_D
    augmented_Y_DATA = Y_DATA
    num_patients = len(X_DATA)

    for numpatient in range(num_patients):

        for shift_perc in shifting_percentages:

            for ax in axes:
            
                shifted_images = shift_tensor(X_DATA[numpatient, :, :, :], shift_param=shift_perc, axis=ax)
                shifted_images = np.expand_dims(shifted_images, axis=0)
                augmented_dataset = np.append(augmented_dataset, shifted_images, axis=0)

                guid = uuid.uuid4()
                current_label = Y_DATA[numpatient]
                                
                augmented_Y_D = np.append(augmented_Y_D, [[str(guid), current_label]], axis=0)
                augmented_Y_DATA = np.append(augmented_Y_DATA, [current_label], axis=0)

    return augmented_dataset, augmented_Y_D, augmented_Y_DATA


def prep_dataset(X_DATA, Y_DATA, num_classes):
    """
    prepare the dataset to be entered in 
    the training pipeline
    """

    X_DATA = X_DATA.astype('float32')
    X_DATA -= np.mean(X_DATA)
    X_DATA /= np.max(X_DATA)
    X_DATA = np.expand_dims(X_DATA, axis=4)#axis to store channel info

    Y_DATA[Y_DATA=='N'] = 0.
    Y_DATA[Y_DATA=='T'] = 0.
    Y_DATA[Y_DATA=='P'] = 1.
    Y_DATA = keras.utils.to_categorical(Y_DATA, num_classes)

    return X_DATA, Y_DATA
    
            
