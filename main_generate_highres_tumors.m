%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Description    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Ziping Liu (2020 Dec)
% Summary:
% This script includes the steps to generate high-resolution simulated tumor images and corresponding background from clinical images

% Patient images containing lung activity but no tumor present were selected as tumor background.
% Pixel size in the high-resolution tumor image was 1/32 of that in the patient image
% Tumor properties, including shape, size and intensity, were sampled from a distribution derived from clinical images 
% Intra-tumor heterogeneity was characterized by a stochastic lumpy object model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define inputs 

export_path             = [];  % Enter export path here

object_size             = 168; % clinical image size
pixel_size              = 4.0728; % Resolution of clinical images
object_fov              = object_size*pixel_size;

%%%%% Define simulated PET system parameters
FWHM                    = 5;
p_num                   = 160;
phi_num                 = 160;

%%%%% Define tumor variabilities obtained from clinical images

% Tumor shape quantified by elliptical fourier descriptors 
fourier_coef            = load('eFourierCoef_PETSegmLung.mat');
efc                     = fourier_coef.eFC_all;

% Tumor size, intensity
patient_metrics         = xlsread('patient_metrics.xls');
tumor_diameter          = patient_metrics(:,8) * 10 / pixel_size;
tumor_volume            = patient_metrics(:,7) * 100 / (pixel_size^2);
tumor_bg_ratio          = patient_metrics(:,6);

bgnd_X_entirefov        = linspace(-object_fov/2, object_fov/2, object_size+1);
bgnd_X_entirefov        = bgnd_X_entirefov(1:object_size);

bgnd_Y_entirefov        = linspace(object_fov/2, -object_fov/2, object_size+1);
bgnd_Y_entirefov        = bgnd_Y_entirefov(1:object_size);

tot_num                 = 100; % Number of total patient images
N_tumor                 = 20;  % Number of tumor locations in each patient image
tot_patient_img         = zeros(168,168,tot_num); % Enter patient images here
tumor_loc_arr           = zeros(N_tumor,2,tot_num); % Enter selected tumor locations here

for iter_ind = 1:tot_num
    
    lung_img = tot_patient_img(:,:,iter_ind);
    
    pre_corrected_lung_sino = gen_subpix_sampling(lung_img,FWHM,object_size,object_fov,p_num,phi_num,1,1);
    ssum = sum(pre_corrected_lung_sino(:));
    scaling_fac = 4e6/ssum; % Scale to clnical count level
    corrected_lung_img = scaling_fac*lung_img;
    
    current_tumor_loc_arr = tumor_loc_arr(:,:,iter_ind);
    
    tumor_centerX_arr = zeros(length(current_tumor_loc_arr),1);
    tumor_centerY_arr = zeros(length(current_tumor_loc_arr),1);
    cropped_bgnd_arr = zeros(object_size,object_size,length(current_tumor_loc_arr));
    margin_img_arr = zeros(1024,1024,length(current_tumor_loc_arr));
    corrected_tumor_img_arr = zeros(1024,1024,length(current_tumor_loc_arr));
    low_res_ground_truth_arr = zeros(object_size,object_size,length(current_tumor_loc_arr));
    
     % Insert tumor at each location in the current lung image
    for loc_ind = 1:length(current_tumor_loc_arr)
        
        current_tumor_pos = current_tumor_loc_arr(loc_ind,:);
        Location_X = floor(current_tumor_pos(1));
        Location_Y = floor(current_tumor_pos(2));
        tumor_center = [bgnd_X_entirefov(Location_X), bgnd_Y_entirefov(Location_Y)];
        tumor_centerX_arr(loc_ind) = bgnd_X_entirefov(Location_X);
        tumor_centerY_arr(loc_ind) = bgnd_Y_entirefov(Location_Y);
        
        [tmp_tumor_img, tmp_tumor_mask, checkpoint] = generate_tumor(efc, tumor_diameter, max_iter, iter_ind, loc_ind, choice_ind);
        if checkpoint == 1
            [tmp_tumor_img, tmp_tumor_mask, checkpoint] = generate_tumor(efc, tumor_diameter, max_iter, iter_ind, loc_ind, choice_ind);
        end
        
        tumor_size = size(tmp_tumor_mask);
        tumor_row = tumor_size(1);
        tumor_col = tumor_size(2);
        tumor_center_row = round(0.5*tumor_row);
        tumor_center_col = round(0.5*tumor_col);
        
        % Create mask for lumpy tumor model
        lumpy_template = zeros(1024); % Here one pixel in the patient image is divided into 32x32 sub-pixels for high-resolution simulated tumor
        for row_ind = 1:tumor_row
            for col_ind = 1:tumor_col
                tmp_row_ind = 512 + row_ind - tumor_center_row;
                tmp_col_ind = 512 + col_ind - tumor_center_col;
                if tmp_tumor_mask(row_ind,col_ind) > 0
                    lumpy_template(tmp_row_ind,tmp_col_ind) = 1;
                end
            end
        end
        
        low_res_tumor_template = zeros(object_size);
        for row_ind = 1:1024
            for col_ind = 1:1024
                val = lumpy_template(row_ind,col_ind);
                if  val> 0
                    orig_bg_loc_row = floor(Location_Y + (row_ind - 512)/32);
                    orig_bg_loc_col = floor(Location_X + (col_ind - 512)/32);
                    low_res_tumor_template(orig_bg_loc_row,orig_bg_loc_col) = 1;
                end
            end
        end
        
        % Generate margin image and mask around the high-resolution tumor model
        margin_img = zeros(1024);
        for row_ind = 1:1024
            for col_ind = 1:1024
                orig_row_ind = floor((row_ind - 512)/32 + Location_Y);
                orig_col_ind = floor((col_ind - 512)/32 + Location_X);
                if low_res_tumor_template(orig_row_ind,orig_col_ind) == 1
                    margin_img(row_ind,col_ind) = corrected_lung_img(orig_row_ind,orig_col_ind);
                end
            end
        end
        margin_img = margin_img.*(1-lumpy_template);
        margin_img_arr(:,:,loc_ind) = margin_img;

        margin_template = (margin_img>0);
        cropped_bgnd = (1-low_res_tumor_template).*corrected_lung_img;
        cropped_bgnd_arr(:,:,loc_ind) = cropped_bgnd;
        
        % for tumor volume <= 57th percentile of patient population
        if(sum(low_res_tumor_template(:))<=prctile(tumor_volume,57)) 

            % Randomly select number of lumps in the simulated tumor volume
            if (10*rand(1)<=5)
                lump_num = 3;
            else
                lump_num = 4;
            end
            TBR = generate_TBR(tumor_bg_ratio);
            while(TBR < 1 || TBR > 15)
                TBR = generate_TBR(tumor_bg_ratio);
            end

            tumor_bgnd_overlap_area_intensity = low_res_tumor_template.*corrected_lung_img;
            tumor_bgnd_overlap_area_intensity = tumor_bgnd_overlap_area_intensity(tumor_bgnd_overlap_area_intensity>0);
            tumor_bgnd_overlap_area_mean_intensity = mean(tumor_bgnd_overlap_area_intensity(:));
            tumor_bgnd_overlap_area_max_intensity = max(tumor_bgnd_overlap_area_intensity(:));
            correct_tumor_mean_intensity = tumor_bgnd_overlap_area_mean_intensity*TBR;

            lump_params = zeros(1,lump_num*4);
            [lumpy_row,lumpy_col] = ind2sub(size(lumpy_template),find(lumpy_template>0));
            min_lumpy_row = min(lumpy_row);
            max_lumpy_row = max(lumpy_row);
            min_lumpy_col = min(lumpy_col);
            max_lumpy_col = max(lumpy_col);

            for lump_ind = 1:lump_num
                [centerX,centerY] = generate_lump_center(min_lumpy_row,max_lumpy_row,min_lumpy_col,max_lumpy_col,lumpy_template);
                min_sigma_val = 1/6*max(max_lumpy_row-min_lumpy_row,max_lumpy_col-min_lumpy_col)*pixel_size/32;
                max_sigma_val = 1/3*max(max_lumpy_row-min_lumpy_row,max_lumpy_col-min_lumpy_col)*pixel_size/32;
                sigma = min_sigma_val + rand(1)*(max_sigma_val-min_sigma_val);
                amp = 5 + rand(1)*(15-5);
                lump_params(1,lump_ind) = amp;
                lump_params(1,1*lump_num+lump_ind) = sigma;
                lump_params(1,2*lump_num+lump_ind) = centerX;
                lump_params(1,3*lump_num+lump_ind) = centerY;
            end

            current_tumor_img = gen_tumor_img(lump_num,lump_params,lumpy_template,tumor_center,1024,1024*pixel_size/32);
            tumor_img_intensity = current_tumor_img(current_tumor_img>0);
            tumor_img_mean_intensity = mean(tumor_img_intensity);
            tumor_amp_scaling_fac = correct_tumor_mean_intensity/tumor_img_mean_intensity;
            lump_params(1:lump_num) = lump_params(1:lump_num).*tumor_amp_scaling_fac;
            corrected_tumor_img = gen_tumor_img(lump_num,lump_params,lumpy_template,tumor_center,1024,1024*pixel_size/32);
            corrected_tumor_img_plus_bgnd = corrected_tumor_img + tumor_bgnd_overlap_area_max_intensity;
            corrected_tumor_img_plus_bgnd = corrected_tumor_img_plus_bgnd.*lumpy_template;

            corrected_tumor_img = corrected_tumor_img_plus_bgnd;
            
        % for tumor volume larger than 57th percentile of patient population
        else
            if (10*rand(1)<=5)
                lump_num = 4;
            else
                lump_num = 5;
            end

            % Create necrotic tumor with 50% probability
            if (10*rand(1)<=5)
                TBR = generate_TBR(tumor_bg_ratio);
                while(TBR < 2 || TBR > 15)
                    TBR = generate_TBR(tumor_bg_ratio);
                end
                tumor_bgnd_overlap_area_intensity = low_res_tumor_template.*corrected_lung_img;
                tumor_bgnd_overlap_area_intensity = tumor_bgnd_overlap_area_intensity(tumor_bgnd_overlap_area_intensity>0);
                tumor_bgnd_overlap_area_mean_intensity = mean(tumor_bgnd_overlap_area_intensity(:));
                tumor_bgnd_overlap_area_max_intensity = max(tumor_bgnd_overlap_area_intensity(:));
                correct_tumor_mean_intensity = tumor_bgnd_overlap_area_mean_intensity*TBR;

                lump_params = zeros(1,lump_num*4);
                [lumpy_row,lumpy_col] = ind2sub(size(lumpy_template),find(lumpy_template>0));
                min_lumpy_row = min(lumpy_row);
                max_lumpy_row = max(lumpy_row);
                min_lumpy_col = min(lumpy_col);
                max_lumpy_col = max(lumpy_col);

                for lump_ind = 1:lump_num
                    [centerX,centerY] = generate_lump_center(min_lumpy_row,max_lumpy_row,min_lumpy_col,max_lumpy_col,lumpy_template);
                    min_sigma_val = 1/6*max(max_lumpy_row-min_lumpy_row,max_lumpy_col-min_lumpy_col)*pixel_size/32;
                    max_sigma_val = 1/3*max(max_lumpy_row-min_lumpy_row,max_lumpy_col-min_lumpy_col)*pixel_size/32;
                    sigma = min_sigma_val + rand(1)*(max_sigma_val-min_sigma_val);
                    amp = 5 + rand(1)*(15-5);
                    lump_params(1,lump_ind) = amp;
                    lump_params(1,1*lump_num+lump_ind) = sigma;
                    lump_params(1,2*lump_num+lump_ind) = centerX;
                    lump_params(1,3*lump_num+lump_ind) = centerY;
                end
                
                current_tumor_img = gen_tumor_img(lump_num,lump_params,lumpy_template,tumor_center,1024,1024*pixel_size/32);
                tumor_img_intensity = current_tumor_img(current_tumor_img>0);
                tumor_img_mean_intensity = mean(tumor_img_intensity);
                tumor_amp_scaling_fac = correct_tumor_mean_intensity/tumor_img_mean_intensity;
                lump_params(1:lump_num) = lump_params(1:lump_num).*tumor_amp_scaling_fac;
                corrected_tumor_img = gen_tumor_img(lump_num,lump_params,lumpy_template,tumor_center,1024,1024*pixel_size/32);
                corrected_tumor_img_plus_bgnd = corrected_tumor_img + tumor_bgnd_overlap_area_max_intensity;
                corrected_tumor_img_plus_bgnd = corrected_tumor_img_plus_bgnd.*lumpy_template;

                corrected_tumor_img = corrected_tumor_img_plus_bgnd;
                
            else
                % Generate necrotic tumor
                disp(['Generating necrotic tumor at loc ',num2str(loc_ind),' choice ',num2str(choice_ind),'..........']);
                
                TBR = generate_TBR(tumor_bg_ratio);
                while(TBR < 2.5 || TBR > 15)
                    TBR = generate_TBR(tumor_bg_ratio);
                end
                tumor_bgnd_overlap_area_intensity = low_res_tumor_template.*corrected_lung_img;
                tumor_bgnd_overlap_area_intensity = tumor_bgnd_overlap_area_intensity(tumor_bgnd_overlap_area_intensity>0);
                tumor_bgnd_overlap_area_mean_intensity = mean(tumor_bgnd_overlap_area_intensity(:));
                tumor_bgnd_overlap_area_max_intensity = max(tumor_bgnd_overlap_area_intensity(:));
                correct_tumor_mean_intensity = tumor_bgnd_overlap_area_mean_intensity*TBR;

                lump_params = zeros(1,lump_num*4);
                [lumpy_row,lumpy_col] = ind2sub(size(lumpy_template),find(lumpy_template>0));

                min_lumpy_row = min(lumpy_row);
                max_lumpy_row = max(lumpy_row);
                min_lumpy_col = min(lumpy_col);
                max_lumpy_col = max(lumpy_col); 
                min_sigma_val = 1/6*max(max_lumpy_row-min_lumpy_row,max_lumpy_col-min_lumpy_col)*pixel_size/32;
                max_sigma_val = 1/3*max(max_lumpy_row-min_lumpy_row,max_lumpy_col-min_lumpy_col)*pixel_size/32;
                for lump_ind = 1:lump_num
                    [centerX,centerY] = generate_lump_center(min_lumpy_row,max_lumpy_row,min_lumpy_col,max_lumpy_col,lumpy_template);      
                    sigma = min_sigma_val + rand(1)*(max_sigma_val-min_sigma_val);
                    amp = 5 + rand(1)*(15-5);
                    lump_params(1,lump_ind) = amp;
                    lump_params(1,1*lump_num+lump_ind) = sigma;
                    lump_params(1,2*lump_num+lump_ind) = centerX;
                    lump_params(1,3*lump_num+lump_ind) = centerY;
                end
                lump_params(1,lump_num) = -1*lump_params(1,lump_num);
                lump_params(1,2*lump_num+lump_num) = min_lumpy_row + floor(0.5*(max_lumpy_row-min_lumpy_row));
                lump_params(1,3*lump_num+lump_num) = min_lumpy_col + floor(0.5*(max_lumpy_col-min_lumpy_col));
                current_tumor_img = gen_tumor_img(lump_num,lump_params,lumpy_template,tumor_center,1024,1024*pixel_size/32);
                min_val = min(current_tumor_img(:));
                if (min_val)<0
                    current_tumor_img = current_tumor_img - min_val;
                    current_tumor_img = current_tumor_img.*lumpy_template;
                end
                tumor_img_intensity = current_tumor_img(current_tumor_img>0);
                tumor_img_mean_intensity = mean(tumor_img_intensity);
                tumor_amp_scaling_fac = correct_tumor_mean_intensity/tumor_img_mean_intensity;
                corrected_tumor_img = current_tumor_img.*tumor_amp_scaling_fac;
                corrected_tumor_img_plus_bgnd = corrected_tumor_img + tumor_bgnd_overlap_area_max_intensity;
                corrected_tumor_img_plus_bgnd = corrected_tumor_img_plus_bgnd.*lumpy_template;

                corrected_tumor_img = corrected_tumor_img_plus_bgnd;

            end
        end
        
        corrected_tumor_img_arr(:,:,loc_ind,choice_ind) = corrected_tumor_img;
        tmp_low_res_tumor_img = zeros(object_size);
        ct = zeros(object_size);
        for row_ind = 1:1024
            for col_ind = 1:1024
                val = corrected_tumor_img(row_ind,col_ind) + margin_img(row_ind,col_ind);
                if  val> 0
                    orig_bg_loc_row = floor(Location_Y + (row_ind - 512)/32);
                    orig_bg_loc_col = floor(Location_X + (col_ind - 512)/32);
                    tmp_low_res_tumor_img(orig_bg_loc_row,orig_bg_loc_col) = tmp_low_res_tumor_img(orig_bg_loc_row,orig_bg_loc_col) + val;
                    ct(orig_bg_loc_row,orig_bg_loc_col) = ct(orig_bg_loc_row,orig_bg_loc_col) + 1;
                end
            end
        end
        tmp_low_res_tumor_img = tmp_low_res_tumor_img./ct;
        tmp_low_res_tumor_img(isnan(tmp_low_res_tumor_img))=0;
        low_res_ground_truth_arr(:,:,loc_ind) = tmp_low_res_tumor_img + cropped_bgnd;
            
    end
    
    save([export_path,'/slice',num2str(iter_ind),'_corrected_lung_img.mat'],'corrected_lung_img');
    save([export_path,'/slice',num2str(iter_ind),'_corrected_tumor_img.mat'],'corrected_tumor_img_arr');
    save([export_path,'/slice',num2str(iter_ind),'_cropped_bgnd_img.mat'],'cropped_bgnd_arr');
    save([export_path,'/slice',num2str(iter_ind),'_margin_img.mat'],'margin_img_arr');
    save([export_path,'/slice',num2str(iter_ind),'_low_res_ground_truth.mat'],'low_res_ground_truth_arr');
    save([export_path,'/slice',num2str(iter_ind),'_tumor_centerX_arr.mat'],'tumor_centerX_arr');
    save([export_path,'/slice',num2str(iter_ind),'_tumor_centerY_arr.mat'],'tumor_centerY_arr');
    
end













