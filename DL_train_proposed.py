###############################################################################
#                            Import Libraries
###############################################################################
import numpy as np
import scipy.io as sio
import tensorflow as tf

from keras import backend as K

K.set_image_data_format('channels_last')
K.tensorflow_backend._get_available_gpus()

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant

config = tf.ConfigProto( device_count = {'GPU': 1} )
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)

###############################################################################
#                            Network Architecture
###############################################################################

#  Define loss function
def loss_fn(y_true, y_pred):
	y_true = tf.stop_gradient(y_true)
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))

# Define accuracy metric
def dice_coef(y_true, y_pred):

	y_pred = tf.nn.softmax(y_pred)
	
	y_pred_1 = y_pred[:,:,:,0]

	y_pred_2 = 1 - y_pred_1
	
	tmp_TP = tf.minimum(y_pred_1,y_true[:,:,:,0])
	TP = tf.reduce_sum(tmp_TP,[1,2])
	tmp_FP = tf.maximum(y_pred_1-y_true[:,:,:,0], 0)
	FP = tf.reduce_sum(tmp_FP,[1,2])
	tmp_FN = tf.maximum(y_pred_2-y_true[:,:,:,1], 0)
	FN = tf.reduce_sum(tmp_FN,[1,2])
	
	nominator = tf.multiply(TP,2)
	tmp_denominator = tf.add(FP,FN)
	denominator = tf.add(tmp_denominator, tf.multiply(TP,2))
	DSC = tf.reduce_mean(tf.divide(nominator,denominator))

	return DSC

def add_conv_layer(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv2D(num_filter, (filter_size, filter_size), # num. of filters and kernel size 
                   strides=stride_size,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = LeakyReLU(alpha=leaky_alpha)(layer) # activation func. 

    return layer

def add_transposed_conv_layer(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
	
    layer = Conv2DTranspose(num_filter, (filter_size, filter_size), # num. of filters and kernel size 
                   strides=stride_size,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer) 
    layer = LeakyReLU(alpha=leaky_alpha)(layer) # activation func. 

    return layer


def cnet(start_filter_num, filter_size, stride_size, dropout_ratio, input_size = (168,168,1)):
	
	input = Input(input_size)
	
	conv1 = add_conv_layer(start_filter_num, filter_size, (1, 1), input) 									
	down1 = add_conv_layer(start_filter_num, filter_size, (stride_size, stride_size), conv1) 				
	
	conv2 = add_conv_layer(start_filter_num*2, filter_size, (1, 1), down1)									
	down2 = add_conv_layer(start_filter_num*2, filter_size, (stride_size, stride_size), conv2)				
	
	conv3 = add_conv_layer(start_filter_num*4, filter_size, (1, 1), down2)									
	down3 = add_conv_layer(start_filter_num*4, filter_size, (stride_size, stride_size), conv3)				

	conv4 = add_conv_layer(start_filter_num*8, filter_size, (1, 1), down3)									
	conv4 = add_conv_layer(start_filter_num*8, filter_size, (1, 1), conv4)									

	drop4 = Dropout(dropout_ratio)(conv4)
	
	up5 = add_transposed_conv_layer(start_filter_num*4, filter_size, (stride_size, stride_size), drop4)		
	up5 = Add()([up5, conv3])
	conv5 = add_conv_layer(start_filter_num*4, filter_size, (1, 1), up5)									
		
	up6 = add_transposed_conv_layer(start_filter_num*2, filter_size, (stride_size, stride_size), conv5)		
	up6 = Add()([up6, conv2])
	conv6 = add_conv_layer(start_filter_num*2, filter_size, (1, 1), up6)									
	
	up7 = add_transposed_conv_layer(start_filter_num*1, filter_size, (stride_size, stride_size), conv6)		
	up7 = Add()([up7, conv1])
	conv7 = add_conv_layer(start_filter_num*1, filter_size, (1, 1), up7)									
	
	conv8 = add_conv_layer(2,filter_size,(1, 1),conv7)
	
	output = conv8
	
	model = Model(inputs=[input], outputs = [output])
	
	model.compile(loss = loss_fn, optimizer = 'adam', metrics = [dice_coef])
	
	return model
    

###############################################################################
#                            	    Training
###############################################################################

export_path = "" # Define export path here

num_train_pats = []
num_val_pats = []
obj_size = 168

TRAIN_X = np.zeros(num_train_pats, obj_size, obj_size, 1) # Define input training PET images here
TRAIN_Y = np.zeros(num_train_pats, obj_size, obj_size, 2) # Define input training ground-truth TFA maps here

val_X = np.zeros(num_val_pats, obj_size, obj_size, 1) # Define input validation PET images here
val_Y = np.zeros(num_val_pats, obj_size, obj_size, 2) # Define input validation ground-truth TFA maps here

feat_num = 32
filter_size = 3
stride_size = 2
dropout_ratio = 0.1

model = cnet(feat_num,filter_size,stride_size,dropout_ratio)

history = model.fit(TRAIN_X,TRAIN_Y,batch_size=10, epochs=100, validation_data=(val_X, val_Y), verbose=1, shuffle=True)

train_loss = history.history['loss']
sio.savemat(export_path+"train_loss.mat",{"train_loss":train_loss})

train_acc = history.history['fuzzy_dice_coef']
sio.savemat(export_path+"train_acc.mat",{"train_acc":train_acc})

val_loss = history.history['val_loss']
sio.savemat(export_path+"val_loss.mat",{"val_loss":val_loss})

val_acc = history.history['val_fuzzy_dice_coef']
sio.savemat(export_path+"val_acc.mat",{"val_acc":val_acc})
