#include "osem_reconstruct.h"
#include <time.h>
/* float* osem_reconstruct(float* system_matrix, float* poiss_lb_sino, int N_recon, int p_num, int phi_num, int num_subset, int num_iter){
	
	int ind_iter;
	int ind_subset;
	
	int int_part;
	int_part = phi_num / num_subset;

	float* lb_recon = (float*)malloc(N_recon*N_recon*sizeof(float));
	int cp_ind;
	for (cp_ind = 0; cp_ind < N_recon*N_recon; cp_ind++){
			
		*(lb_recon + cp_ind) = 1;
			
	}
	
	float* sinogram_sub = (float*)malloc(p_num*(int_part+1)*sizeof(float));
	float* forward_projection = (float*)malloc(p_num*phi_num*sizeof(float));
	float* fp_sub = (float*)malloc(p_num*(int_part+1)*sizeof(float));
	float* comparison = (float*)malloc(p_num*(int_part+1)*sizeof(float));
	float* H_sub = (float*)malloc(p_num*(int_part+1)*sizeof(float));
	float* back_projection = (float*)malloc(p_num*(int_part+1)*sizeof(float));
	
	int ind_mmp_1;
	int ind_mmp_2;
	float mmp_ele_1;
	float mmp_ele_2;
	float mmp_result;
	
	int cap;
	int ind_cap;
	int ind_n;
	int ind_p;
	float poiss_sino_col;
	float forward_proj_col;
	
	float nom;
	float denom;
	int ind_cap_2;
	int ind_p_2;
	float system_matrix_col;
	
	float normalize_factor;
	float comparison_col;
	float H_sub_col;
	float ssum;
    float tmp;
	
	clock_t begin, end;
	int copy_ind;
	double time_elapsed;
	
	for (ind_iter = 0; ind_iter < num_iter; ind_iter ++){
		
		for (ind_subset = 0; ind_subset < num_subset; ind_subset ++){
			
			
			printf("%d\n",ind_subset);
			printf("Begin timing...");
			begin = clock();
			
			//Initialize  
			for (copy_ind = 0; copy_ind < p_num*phi_num; copy_ind ++){
				*(forward_projection + copy_ind) = 0;
			}
			
			//Calculate estimated sinogram
			for (ind_mmp_1 = 0; ind_mmp_1 < p_num*phi_num; ind_mmp_1++){
				for (ind_mmp_2 = 0; ind_mmp_2 < N_recon*N_recon; ind_mmp_2++){
					mmp_ele_1 = *(system_matrix + ind_mmp_2*p_num*phi_num + ind_mmp_1);
					mmp_ele_2 = *(lb_recon + ind_mmp_2);
					mmp_result = *(forward_projection + ind_mmp_1) + mmp_ele_1 * mmp_ele_2;
					*(forward_projection + ind_mmp_1) = mmp_result;				
				}
			}
			
			
			if (int_part*num_subset+ind_subset < phi_num){
				cap = int_part + 1;
			}
			else{
				cap = int_part;
			}
			
			
			for (ind_cap = 0; ind_cap < cap; ind_cap++){
				for (ind_p = 0; ind_p < p_num; ind_p++){
					poiss_sino_col = *(poiss_lb_sino + p_num*ind_subset + p_num*ind_cap*num_subset + ind_p);
					*(sinogram_sub + ind_cap*p_num + ind_p) = poiss_sino_col;
					forward_proj_col = *(forward_projection + p_num*ind_subset + p_num*ind_cap*num_subset + ind_p);
					*(fp_sub + ind_cap*p_num + ind_p) = forward_proj_col;
					if (forward_proj_col == 0){
						*(comparison + ind_cap*p_num + ind_p) = 0;
					}
					else{
						nom = *(sinogram_sub + ind_cap*p_num + ind_p);
						denom = *(fp_sub + ind_cap*p_num + ind_p);
						*(comparison + ind_cap*p_num + ind_p) = nom/denom;
					}	
				}	
			}
			
			
			for (ind_n = 0; ind_n < N_recon*N_recon; ind_n ++){
			
				normalize_factor = 0;
				ssum = 0;
				
				for (ind_cap_2 = 0; ind_cap_2 < cap; ind_cap_2++){
					for (ind_p_2 = 0; ind_p_2 < p_num; ind_p_2++){ 
						system_matrix_col = *(system_matrix + ind_n*p_num*phi_num + p_num*ind_subset + p_num*ind_cap_2*num_subset + ind_p_2);
						*(H_sub + ind_cap_2*p_num + ind_p_2) = system_matrix_col;
						normalize_factor = normalize_factor + *(H_sub + ind_cap_2*p_num + ind_p_2);
						
						
					}
				}
				
				for (ind_cap_2 = 0; ind_cap_2 < cap; ind_cap_2++){
					for (ind_p_2 = 0; ind_p_2 < p_num; ind_p_2++){ 
					
						if (normalize_factor == 0){
							*(back_projection + ind_cap_2*p_num + ind_p_2) = 0;
						}
						else{
							comparison_col = *(comparison + ind_cap_2*p_num + ind_p_2);
							H_sub_col = *(H_sub + ind_cap_2*p_num + ind_p_2);
							*(back_projection + ind_cap_2*p_num + ind_p_2) =  comparison_col / normalize_factor * H_sub_col;
						}
						ssum = ssum + *(back_projection + ind_cap_2*p_num + ind_p_2);
					}
				}	
			
                tmp = *(lb_recon+ind_n)*ssum;	
				*(lb_recon+ind_n) = tmp;
		
			}

			end = clock();
			time_elapsed = (double)(end-begin)/ CLOCKS_PER_SEC;
			printf("Time takes: %.5f\n",time_elapsed);			
			
			
		}
		
	}
	
	return lb_recon;
	
	
} */



float* osem_reconstruct(float * sinogram, float * H_matrix, int p_num, int phi_num, int N, int subset_num, int iter_num){	
	int ind_iter_main,ind_iter_subset,ind_sbH,cur_num_phi_sub,ind_H_sub_row;
	int ii,jj;
	int ind_H_row, ind_H ;
	int num_phi_sub = ceil((double)phi_num / subset_num);//
	float debug_val;
	
	int * subset_ind_H = (int*)malloc(sizeof(int)*p_num*(num_phi_sub));
	float * fp_subset = (float*)malloc(sizeof(float)*p_num*(num_phi_sub));
	float * weight_data = (float*)malloc(sizeof(float)*p_num*(num_phi_sub));
	float* f_recon = (float*)malloc(N*N*sizeof(float));
	float* sum_Hw = (float*)malloc(N*N*sizeof(float));
	float* sum_H = (float*)malloc(N*N*sizeof(float));
	
	//clock_t begin, end;
	//double time_elapsed;
	
	for (ii = 0; ii < N*N; ii++)
	{			
		f_recon[ii] = 1.0;		
	}
	
	for (ind_iter_main = 0;ind_iter_main<iter_num;ind_iter_main++)
	{
		for (ind_iter_subset = 0;ind_iter_subset<subset_num;ind_iter_subset++)	
		{
			//begin = clock();
			
			memset(subset_ind_H,0,sizeof(int)*p_num*(num_phi_sub));
			memset(fp_subset,0,sizeof(float)*p_num*(num_phi_sub));
			memset(sum_Hw,0,sizeof(float)*N*N);
			memset(sum_H,0,sizeof(float)*N*N);
			memset(weight_data,0,sizeof(float)*p_num*(num_phi_sub));
			
			// part-1: calculate subset indexes of main H matrix (rows)
			cur_num_phi_sub = 0;//number of angles in a subset
			for (ii = 0;ii<(num_phi_sub);ii++)
			{
				for (jj = 0;jj<p_num;jj++)
				{
					ind_H_sub_row = ii * p_num + jj;
					ind_H_row = ind_iter_subset * p_num + (ii * subset_num) * p_num + jj;
					if (ind_H_row < p_num*phi_num){
						subset_ind_H[ind_H_sub_row] = ind_H_row;
						cur_num_phi_sub = ii + 1;	
						//printf("%d||",ind_H_row);
					}
				}
			}
			//printf("\n");
			// part-2: forward projection and weight calculation
			
			
			for (ii = 0; ii<p_num*(cur_num_phi_sub);ii++)
			{
				ind_H_row = subset_ind_H[ii];
				debug_val = 0;
				for (jj = 0; jj< N*N;jj++)
				{
					ind_H = jj * p_num * phi_num + ind_H_row;
					fp_subset[ii] = fp_subset[ii] + H_matrix[ind_H] * f_recon[jj];
					debug_val = debug_val + H_matrix[ind_H];
				}
				//printf("%f||",debug_val);
				//if(fabs(fp_subset[ii])>1e-16){
				if(fp_subset[ii]!=0){
					weight_data[ii] = sinogram[ind_H_row]/fp_subset[ii];}
				else{
					weight_data[ii] = 0;}
			}
			
			// part-3: sensitivity denom
			
			
			// part-4: calculate \sum{H*weight}
			for(jj=0;jj<N*N;jj++)
			{
				for(ii=0;ii<p_num*(cur_num_phi_sub);ii++)
				{
					ind_H_row = subset_ind_H[ii];
					ind_H = jj * p_num * phi_num + ind_H_row; 
					sum_Hw[jj] = sum_Hw[jj] + weight_data[ii] * H_matrix[ind_H]; 
					sum_H [jj] = sum_H [jj] + H_matrix[ind_H];
				}
				if (sum_H [jj]!=0){
					f_recon[jj] = f_recon [jj] * sum_Hw[jj] / sum_H [jj];
				}
				else{
					f_recon[jj] = 0;
				}
			}
			//end = clock();
			//time_elapsed = (double)(end-begin)/ CLOCKS_PER_SEC;
			//printf("iter:%d||cur:%d||max:%d||Time takes: %.5f\n",ind_iter_subset,cur_num_phi_sub,num_phi_sub,time_elapsed);				
		}
	}
	return f_recon;
	
}