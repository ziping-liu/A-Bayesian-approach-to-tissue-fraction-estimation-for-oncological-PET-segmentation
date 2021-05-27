# A-Bayesian-approach-to-tissue-fraction-estimation-for-oncological-PET-segmentation
Open-source code for the proposed tissue-fraction estimation method for oncological PET segmentation

Evaluation of the proposed method using clinically realistic simulation studies:

This folder contains the following main scripts:

1) main_generate_highres_tumors.m

This file contains the MATLAB code to generate high-resolution tumor images with corresponding low-resolution background image obtained from clinical images

2) main_generate_DL_train_ground_truth.m

This file contains the MATLAB code to generate ground-truth tumor-fraction area maps based on the above-generated high-resolution tumor images

3) main_generate_simulated_PET_images.c

This file contains the C code to first generate forward projection of high-resolution tumor images and low-resolution background images, respectively, using simulated PET systems.
The forward projections were then added with incorporation of Poisson noise to feed into OSEM reconstruction algorithm to generate eventual simulated PET images.

4) DL_train_proposed.py

This file contains the Python code to construct en encoder-decoder network for training the proposed method

5) DL_predict_proposed.py

This file contains the Python code to apply the trained network to predict on test images
