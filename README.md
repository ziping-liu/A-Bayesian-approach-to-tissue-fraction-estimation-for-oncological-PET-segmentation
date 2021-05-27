# A-Bayesian-approach-to-tissue-fraction-estimation-for-oncological-PET-segmentation
Open-source code for the proposed tissue-fraction estimation method for oncological PET segmentation

Evaluation of the proposed method using clinical images from multi-center data:

This folder contains the following main scripts:

1) main_prepare_DL_training_data

This file contains the MATLAB code to extract PET images and high-resolution manual segmentations from MIM softward. The high-resolution manual segmentations were then used to generate the ground-truth tumor-fraction area maps.

2) DL_train_proposed.py

This file contains the Python code to construct en encoder-decoder network for training the proposed method

3) DL_predict_proposed.py

This file contains the Python code to apply the trained network to predict on test images
