#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Method to generate sinogram for high-resolution tumor images 
void* gen_tumor_sub_pix_samp_sino(float* tumor_sino, float* tumor_img, int obj_size, int sub_col_num, int sub_row_num, float fwhm, int sino_p_num, int sino_phi_num, float fov, float tumor_fov, float pi, float centerX, float centerY);