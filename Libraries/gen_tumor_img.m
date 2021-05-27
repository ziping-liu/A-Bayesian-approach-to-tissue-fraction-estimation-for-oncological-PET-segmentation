function tumor_img = gen_tumor_img(lump_num,params,lumpy_template,center,tumor_size,tumor_fov)

X_entirefov     = linspace(center(1) - tumor_fov/2, center(1) + tumor_fov/2, tumor_size);
Y_entirefov     = linspace(center(2) + tumor_fov/2, center(2) - tumor_fov/2, tumor_size);

[row,col]       = ind2sub(size(lumpy_template),find(lumpy_template>0));
len             = length(col);

lump_amp_arr = params(1:lump_num);
lump_sigma_arr = params(lump_num+1:2*lump_num);
lump_posX_arr = X_entirefov(params(2*lump_num+1:3*lump_num));
lump_posY_arr = Y_entirefov(params(3*lump_num+1:4*lump_num));
tumor_img = zeros(tumor_size);
for row_ind = 1:size(lumpy_template,1)
    for col_ind = 1:size(lumpy_template,2)
        current_X_val = X_entirefov(col_ind);
        current_Y_val = Y_entirefov(row_ind);
        const_exp_nom = (current_X_val - lump_posX_arr).^2 + (current_Y_val - lump_posY_arr).^2;
        mag = lump_amp_arr.*exp(-1.*const_exp_nom./(2.*(lump_sigma_arr.^2)));
        tot_current_mag = sum(mag(:));
        tumor_img(row_ind,col_ind) = tot_current_mag;
    end
end

angle = unifrnd(0,360);
output_img = imrotate(tumor_img, angle,'crop');
tumor_img = output_img.* lumpy_template;


% for ind = 1:len
%     true_col_ind = col(ind);
%     true_row_ind = row(ind);
%     current_X_val = X_entirefov(true_col_ind);
%     current_Y_val = Y_entirefov(true_row_ind);
%     const_exp_nom = (current_X_val - lump_posX_arr).^2 + (current_Y_val - lump_posY_arr).^2;
%     mag = lump_amp_arr.*exp(-1.*const_exp_nom./(2.*(lump_sigma_arr.^2)));
%     tot_current_mag = sum(mag(:));
%     tumor_img(true_row_ind,true_col_ind) = tot_current_mag;
% end


end