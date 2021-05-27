#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Method to generate sinogram for low-resolution patient images 
void* gen_sub_pix_samp_sino(float* bgnd_sino, float* bgnd_img, int bgnd_size, float bgnd_fov, float pi, float fwhm, int sino_p_num, int sino_phi_num, int sub_col_num, int sub_row_num);