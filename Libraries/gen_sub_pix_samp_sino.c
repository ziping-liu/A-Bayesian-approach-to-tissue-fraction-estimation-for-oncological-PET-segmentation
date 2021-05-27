#include "gen_sub_pix_samp_sino.h"

void* gen_sub_pix_samp_sino(float* bgnd_sino, float* bgnd_img, int bgnd_size, float bgnd_fov, float pi, float fwhm, int sino_p_num, int sino_phi_num, int sub_col_num, int sub_row_num){
	
	float sigma_det 	= fwhm/(2*sqrt(2*log(2)));
	
	float* X_arr = (float*)malloc(bgnd_size*sub_col_num*sizeof(float));
	float* Y_arr = (float*)malloc(bgnd_size*sub_row_num*sizeof(float));
	
	float p_min			= -0.5*bgnd_fov;
	float p_max 		= 0.5*bgnd_fov;
	float p_res			= (p_max-p_min)/(float)sino_p_num;

	float* theta_val = (float*)malloc(sino_phi_num*sizeof(float));
	
	int theta_ind;
	float mu;
	int p_ind;
	
	int col_ind;
	int sub_col_ind;
	int row_ind;
	int sub_row_ind;
	int arr_ind;
	int bgnd_col_ind;
	int bgnd_row_ind;
	
	int* col_arr = (int*)malloc(bgnd_size*bgnd_size*sizeof(int));
	int* row_arr = (int*)malloc(bgnd_size*bgnd_size*sizeof(int));
	
	float x_value;
	float y_value;
	float p_value;
	int p_cent_ind;
	int no_pixel_considered = ceil(3*sigma_det/p_res);
	float ratio;
	
	float cdf_start;
	float cdf_end;
	float cdf1;
	float cdf2;
	
	float scaling_factor = bgnd_fov*bgnd_fov/((float)bgnd_size*(float)bgnd_size*(float)sub_row_num*(float)sub_col_num);
	
	int i;
	for (i=0;i<sino_phi_num;i++){
		*(theta_val+i) = pi * (float)(i)/((float)sino_phi_num);
	}
	for (i=0;i<bgnd_size*sub_col_num;i++){
		*(X_arr+i) = -1.0*0.5*bgnd_fov + bgnd_fov*((float)i)/((float)bgnd_size*sub_col_num - 1.0);
		*(Y_arr+i) = 0.5*bgnd_fov - bgnd_fov*((float)i)/((float)bgnd_size*sub_row_num - 1.0);
	}
	
	float val;
	int count = 0;
	for (col_ind = 0; col_ind < bgnd_size; col_ind++){
		for (row_ind = 0; row_ind < bgnd_size; row_ind++){
			val = *(bgnd_img + col_ind*bgnd_size + row_ind);
			if (val > 0){
				*(col_arr + count) = col_ind;
				*(row_arr + count) = row_ind;
				count = count + 1;
			}
		}
	}
	//printf("count = %d\n",count);
				
	float mag;
	for (theta_ind = 0; theta_ind < sino_phi_num; theta_ind++){
		
		for (arr_ind = 0; arr_ind < count; arr_ind ++){
	
			bgnd_col_ind = *(col_arr + arr_ind);
			bgnd_row_ind = *(row_arr + arr_ind);
			mag = *(bgnd_img + bgnd_col_ind*bgnd_size + bgnd_row_ind);
			
			for (sub_col_ind = 0; sub_col_ind < sub_col_num; sub_col_ind++){
				
				for (sub_row_ind = 0; sub_row_ind < sub_row_num; sub_row_ind++){
					
					x_value = *(X_arr + bgnd_col_ind*sub_col_num + sub_col_ind);
					y_value = *(Y_arr + bgnd_row_ind*sub_row_num + sub_row_ind);

					p_value = x_value*cos(*(theta_val+theta_ind)) + y_value*sin(*(theta_val+theta_ind));
					mu = p_value;
					
					if (p_value >= p_min && p_value <= p_max){
						
						p_cent_ind = floor((mu - p_min)/(p_max - p_min)*sino_p_num);
						
						for (p_ind=(p_cent_ind-no_pixel_considered); p_ind<(p_cent_ind+no_pixel_considered+1); p_ind++){
							
							if (p_ind >=0 && p_ind < sino_p_num){
								
								cdf_start = p_min + p_ind*p_res;
								cdf_end = cdf_start + p_res;
								cdf1 = 0.5*(1+erf((cdf_start-mu)/(sigma_det*sqrt(2))));
								cdf2 = 0.5*(1+erf((cdf_end-mu)/(sigma_det*sqrt(2))));
								ratio = cdf2 - cdf1;
								*(bgnd_sino + theta_ind*sino_p_num + p_ind) = *(bgnd_sino + theta_ind*sino_p_num + p_ind) + ratio*mag*scaling_factor;	
							
							}
							
						}
						
					}		
					
				}
				
			}
				
		}
		
		//printf("%d / %d\n",theta_ind,sino_phi_num);
		
	}
	
}