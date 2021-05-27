###############################################################################
#                            Import Libraries
###############################################################################

import numpy as np
import scipy.io as sio

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import load_model
    
export_path = "" # Define export path here

num_test_pats = []
obj_size = 168

test_X = np.zeros(num_test_pats, obj_size, obj_size, 1) # Define input test PET images here

proposed_model = load_model("proposed_model.hdf5",compile=False)

proposed_pred = proposed_model.predict(test_X)
proposed_pred = K.eval(K.softmax(proposed_pred))
    
sio.savemat(export_path+"/proposed_pred.mat",{"proposed_pred":proposed_pred})