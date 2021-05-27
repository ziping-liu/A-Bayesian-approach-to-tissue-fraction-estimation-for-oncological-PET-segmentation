%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Description    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Ziping Liu (2020 Dec)
% Summary:
% This script includes the steps to generate ground-truth tumor-fraction area map from the high-resolution tumor images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Position of tumor center in the clinical image to be inserted
tumor_posX              = [];
tumor_posY              = [];

object_size             = 168; % clinical image size
pixel_size              = 4.0728; % Resolution of clinical images
high_res_pixel_size     = pixel_size/32;

object_fov              = object_size*pixel_size;
tumor_fov               = high_res_pixel_size*1024;

bgnd_X_entirefov = linspace(-object_fov/2, object_fov/2, object_size+1);
bgnd_X_entirefov = bgnd_X_entirefov(1:object_size);

bgnd_Y_entirefov = linspace(object_fov/2, -object_fov/2, object_size+1);
bgnd_Y_entirefov = bgnd_Y_entirefov(1:object_size);

Location_Y = find(abs(bgnd_Y_entirefov-tumor_posY)<0.001);
Location_X = find(abs(bgnd_X_entirefov-tumor_posX)<0.001);			

bgnd_X_tumorfov = linspace(tumor_posX-tumor_fov/2, tumor_posX+tumor_fov/2, 1024+1);
bgnd_X_tumorfov = bgnd_X_tumorfov(1:1024);

bgnd_Y_tumorfov = linspace(tumor_posY+tumor_fov/2, tumor_posY-tumor_fov/2, 1024+1);
bgnd_Y_tumorfov = bgnd_Y_tumorfov(1:1024);

cont_gt_tot_ct = zeros(object_size);
cont_gt = zeros(object_size);
for row_ind = 1:1024
    for col_ind = 1:1024
        val = final_corrected_tumor_img(row_ind,col_ind);
        current_X = bgnd_X_tumorfov(col_ind);
        current_Y = bgnd_Y_tumorfov(row_ind);
        orig_bg_loc_row = find(current_Y-bgnd_Y_entirefov>0);
        orig_bg_loc_row = orig_bg_loc_row(1)-1;
        orig_bg_loc_col = find(bgnd_X_entirefov-current_X>0);
        orig_bg_loc_col = orig_bg_loc_col(1)-1;
        cont_gt_tot_ct(orig_bg_loc_row,orig_bg_loc_col) = cont_gt_tot_ct(orig_bg_loc_row,orig_bg_loc_col) + 1;
        if val>0
            cont_gt(orig_bg_loc_row,orig_bg_loc_col) = cont_gt(orig_bg_loc_row,orig_bg_loc_col) + 1;
        end
    end
end
cont_gt = cont_gt./cont_gt_tot_ct;
cont_gt(isnan(cont_gt)) = 0;

