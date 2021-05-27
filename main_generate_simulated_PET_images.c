#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gen_poisson_rv.h"
#include "gen_sub_pix_samp_sino.h"
#include "gen_tumor_sub_pix_samp_sino.h"
#include "osem_reconstruct.h"

int main(int argc ,char **argv)
{

	int start_ID = atoi(argv[1]);
	
	int end_ID = atoi(argv[2]);
	
	int simul_ID = atoi(argv[3]);
	
	simul_ID = simul_ID-1;
	printf("start_ID=%d\n",start_ID);
	printf("end_ID=%d\n",end_ID);
	printf("simul_ID=%d\n",simul_ID);
	
	float PI 					= 3.1415926;
	int P_NUM 					= 160;
	int PHI_NUM 				= 160;
	
	int LUNG_WIDTH 				= 168;
	int LUNG_HEIGHT 			= 168;
	
	int TUMOR_WIDTH				= 1024;
	int TUMOR_HEIGHT			= 1024;
	
	float FWHM 					= 5;
	
	float FOV					= 684.2304;
	float TUMOR_FOV				= 130.3296;
	
	int N_recon 				= 168;
	int num_subset 				= 21;
	int num_iter				= 2;
	
	int TOT_NUM					= 180;
	
	// Read System matrix data
	float* system_matrix = (float*)malloc(PHI_NUM*P_NUM*N_recon*N_recon*sizeof(float));
	FILE* system_matrix_data;
	char filename_system_matrix[64];
	sprintf(filename_system_matrix, "/scratch/liuziping/system_matrix_folder/system_matrix_fov_684_p_160_phi_160.dat");
	system_matrix_data = fopen(filename_system_matrix,"rb");
	if(system_matrix_data == NULL){	
		printf("Error in opening system matrix file \n");
	}
	fread(system_matrix, sizeof(float), PHI_NUM*P_NUM*N_recon*N_recon, system_matrix_data);
	fclose(system_matrix_data);
	printf("system_matrix data loaded \n");
	
	// Load cropped lung data 
	float* lung_data = (float*)malloc(LUNG_WIDTH*LUNG_HEIGHT*TOT_NUM*sizeof(float));
	FILE* file_lung_data;
	char filename_lung_data[128];
	sprintf(filename_lung_data, "/scratch/liuziping/patient_imgs_168/corrected_cropped_bgnd_arr_%d.dat",simul_ID);	
	file_lung_data = fopen(filename_lung_data,"rb");
	if(file_lung_data == NULL){
		printf("Error in opening lung data file\n");	
	}
	fread(lung_data, sizeof(float), LUNG_WIDTH*LUNG_HEIGHT*TOT_NUM, file_lung_data);
	fclose(file_lung_data);	
	printf("Lung data loaded \n");
	
	// Load tumor data
	float* tumor_img = (float*)malloc(TUMOR_WIDTH*TUMOR_HEIGHT*TOT_NUM*sizeof(float));
	FILE* file_tumor_img;
	char filename_tumor_img[128];
	sprintf(filename_tumor_img, "/scratch/liuziping/patient_imgs_168/corrected_tumor_arr_%d.dat",simul_ID);
	file_tumor_img = fopen(filename_tumor_img,"rb");
	if(file_tumor_img == NULL){	
		printf("Error in opening tumor data file\n");	
	}
	fread(tumor_img, sizeof(float), TUMOR_WIDTH*TUMOR_HEIGHT*TOT_NUM, file_tumor_img);
	fclose(file_tumor_img);
	printf("Tumor data loaded \n");
	
	// Load margin data
	float* margin_img = (float*)malloc(TUMOR_WIDTH*TUMOR_HEIGHT*TOT_NUM*sizeof(float));
	FILE* file_margin_img;
	char filename_margin_img[128];
	sprintf(filename_margin_img, "/scratch/liuziping/patient_imgs_168/corrected_margin_arr_%d.dat",simul_ID);
	file_margin_img = fopen(filename_margin_img,"rb");
	if(file_margin_img == NULL){	
		printf("Error in opening margin data file\n");	
	}
	fread(margin_img, sizeof(float), TUMOR_WIDTH*TUMOR_HEIGHT*TOT_NUM, file_margin_img);
	fclose(file_margin_img);
	printf("Margin data loaded \n");
	
	// Load tumor pos X data
	float* tumor_pos_X_arr = (float*)malloc(TOT_NUM*sizeof(float));
	FILE* file_tumor_pos_X_arr;
	char filename_tumor_pos_X_arr[64];
	sprintf(filename_tumor_pos_X_arr, "/scratch/liuziping/patient_imgs_168/corrected_tumor_pos_arrX_%d.dat",simul_ID);
	file_tumor_pos_X_arr = fopen(filename_tumor_pos_X_arr,"rb");
	if(file_tumor_pos_X_arr == NULL){	
		printf("Error in opening tumor pos X arr file\n");	
	}
	fread(tumor_pos_X_arr, sizeof(float), TOT_NUM, file_tumor_pos_X_arr);
	fclose(file_tumor_pos_X_arr);
	printf("Tumor pos X data loaded \n");
	
	// Load tumor pos Y data
	float* tumor_pos_Y_arr = (float*)malloc(TOT_NUM*sizeof(float));
	FILE* file_tumor_pos_Y_arr;
	char filename_tumor_pos_Y_arr[64];
	sprintf(filename_tumor_pos_Y_arr, "/scratch/liuziping/patient_imgs_168/corrected_tumor_pos_arrY_%d.dat",simul_ID);
	file_tumor_pos_Y_arr = fopen(filename_tumor_pos_Y_arr,"rb");
	if(file_tumor_pos_Y_arr == NULL){	
		printf("Error in opening tumor pos Y arr file\n");	
	}
	fread(tumor_pos_Y_arr, sizeof(float), TOT_NUM, file_tumor_pos_Y_arr);
	fclose(file_tumor_pos_Y_arr);
	printf("Tumor pos Y data loaded \n");
	
	float* current_lung_sino			= (float*)malloc(PHI_NUM*P_NUM*sizeof(float));
	float* current_tumor_margin_sino	= (float*)malloc(PHI_NUM*P_NUM*sizeof(float));
	
	float combined_sino_intensity;
	
	float* combined_poiss_sino			= (float*)malloc(PHI_NUM*P_NUM*sizeof(float));
	float* combined_recon				= (float*)malloc(N_recon*N_recon*sizeof(float));
	
	int iter_num;
	printf("%d\n",start_ID);
	printf("%d\n",end_ID);
	
	for (iter_num=start_ID; iter_num<end_ID;iter_num++){
		
		float* current_lung_img				= (float*)malloc(LUNG_WIDTH*LUNG_HEIGHT*sizeof(float));
		float* current_lung_sino			= (float*)malloc(PHI_NUM*P_NUM*sizeof(float));
		
		float* current_tumor_margin_img		= (float*)malloc(TUMOR_WIDTH*TUMOR_HEIGHT*sizeof(float));
		float* current_tumor_margin_sino	= (float*)malloc(PHI_NUM*P_NUM*sizeof(float));
		
		float combined_sino_intensity;
		float* combined_poiss_sino			= (float*)malloc(PHI_NUM*P_NUM*sizeof(float));
		float* combined_recon				= (float*)malloc(N_recon*N_recon*sizeof(float));
		
		float current_centerX; 
		float current_centerY;
		
		int ind;
		for (ind = 0; ind < LUNG_WIDTH*LUNG_HEIGHT; ind++){
			*(current_lung_img + ind) = *(lung_data + ind + iter_num*LUNG_WIDTH*LUNG_HEIGHT);
		}
		
		for (ind = 0; ind < TUMOR_WIDTH*TUMOR_HEIGHT; ind++){
			*(current_tumor_margin_img + ind) = *(tumor_img + ind + iter_num*TUMOR_WIDTH*TUMOR_HEIGHT) + *(margin_img + ind + iter_num*TUMOR_WIDTH*TUMOR_HEIGHT);
		}
		
		for (ind = 0; ind < PHI_NUM*P_NUM; ind++){
			*(current_tumor_margin_sino + ind) = 0;
			*(current_lung_sino + ind) = 0;
		}
			
		printf("iter_num=%d\n",iter_num);
		
		char filename_lung_sino[128];
		sprintf(filename_lung_sino, "/scratch/liuziping/out_patient_imgs_168/%d/lung_sino%d.dat",simul_ID,iter_num);
		FILE* lung_sinoOUT;
		lung_sinoOUT = fopen(filename_lung_sino,"wb");
		if (lung_sinoOUT == NULL){
			printf("Error opening lung sino output file!\n");
		}
		gen_sub_pix_samp_sino(current_lung_sino, current_lung_img, LUNG_WIDTH, FOV, PI, FWHM, P_NUM, PHI_NUM, 2, 2);
		fwrite(current_lung_sino, sizeof(float), P_NUM*PHI_NUM, lung_sinoOUT);
		fclose(lung_sinoOUT);	
		
		char filename_tumor_margin_sino[128];
		sprintf(filename_tumor_margin_sino, "/scratch/liuziping/out_patient_imgs_168/%d/tumor_margin_sino%d.dat",simul_ID,iter_num);
		FILE* tumor_margin_sinoOUT;
		tumor_margin_sinoOUT = fopen(filename_tumor_margin_sino,"wb");
		if (tumor_margin_sinoOUT == NULL){
			printf("Error opening tumor margin sino output file!\n");
		}
		current_centerX = *(tumor_pos_X_arr + iter_num);
		current_centerY = *(tumor_pos_Y_arr + iter_num);
		gen_tumor_sub_pix_samp_sino(current_tumor_margin_sino, current_tumor_margin_img, TUMOR_WIDTH, 1, 1, FWHM, P_NUM, PHI_NUM, FOV, TUMOR_FOV, PI, current_centerX, current_centerY);
		fwrite(current_tumor_margin_sino, sizeof(float), P_NUM*PHI_NUM, tumor_margin_sinoOUT);
		fclose(tumor_margin_sinoOUT);
		
		for (ind = 0; ind < PHI_NUM*P_NUM; ind++){
			
			combined_sino_intensity = *(current_lung_sino + ind) + *(current_tumor_margin_sino + ind);
	
			if(combined_sino_intensity < 0){
				*(combined_poiss_sino + ind) = 0;
			}
			else{
				*(combined_poiss_sino + ind) = gen_poisson_rv(combined_sino_intensity);
			}
		
		}
		
		char filename_combined_recon[128];
		sprintf(filename_combined_recon, "/scratch/liuziping/out_patient_imgs_168/%d/combined_recon%d.dat",simul_ID,iter_num);
		FILE* combined_reconOUT;
		combined_reconOUT = fopen(filename_combined_recon,"wb");
		if (combined_reconOUT == NULL){
			printf("Error opening combined recon output file!\n");
		}
		combined_recon = osem_reconstruct(combined_poiss_sino, system_matrix, P_NUM, PHI_NUM, N_recon, num_subset, num_iter);
		fwrite(combined_recon, sizeof(float), N_recon*N_recon, combined_reconOUT);
		fclose(combined_reconOUT);	
			
	}		
	
	free(combined_recon);
	free(combined_poiss_sino);
	free(current_tumor_margin_sino);
	free(current_lung_sino);	
	free(tumor_pos_X_arr);
	free(tumor_pos_Y_arr);
	free(margin_img);
	free(tumor_img);
	free(lung_data);
	free(system_matrix);
}
