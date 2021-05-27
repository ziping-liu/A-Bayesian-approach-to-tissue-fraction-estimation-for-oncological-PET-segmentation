function [sinogram] = gen_subpix_sampling(img,FWHM,obj_size,fov,p_num,phi_num,sub_row_num,sub_col_num)
 
% Find the standard deviation of the gaussian distribution 
sigma_det       = FWHM/(2*sqrt(2*log(2)));
 
X_entirefov     = linspace(-fov/2, fov/2, obj_size*sub_col_num);
Y_entirefov     = linspace(fov/2, -fov/2, obj_size*sub_row_num);
 
theta_arr       = linspace(0,pi,phi_num+1);
theta_arr       = theta_arr(1:phi_num);
 
p_min           = -1/2*fov;
p_max           = 1/2*fov;
p_res           = (p_max-p_min)/p_num;
 
no_pixels_considered = ceil(3*sigma_det/p_res);
 
p_arr           = linspace(p_min, p_max, p_num+1);
p_arr           = p_arr(1:p_num);
 
[row,col] = ind2sub(size(img),find(img>0));
 
len = length(col);
 
scaling_factor = fov^2/(obj_size^2*sub_row_num*sub_col_num);
 
sinogram = cast(zeros(p_num,phi_num),'double');
 
for theta_ind = 1:phi_num
    for ind = 1:len
        true_col_ind = col(ind);
        true_row_ind = row(ind);
        mag = img(true_row_ind, true_col_ind);
        for sub_col_ind = 1:sub_col_num
            for sub_row_ind = 1:sub_row_num 
                x_val = X_entirefov(sub_col_num*(true_col_ind-1)+sub_col_ind);
                y_val = Y_entirefov(sub_row_num*(true_row_ind-1)+sub_row_ind);
 
                p_val = x_val*cos(theta_arr(theta_ind))+y_val*sin(theta_arr(theta_ind));  
                if p_val>=p_min && p_val<= p_max
                    p_cent_ind = floor((p_val - p_min)/(p_max-p_min)*p_num)+1;
                    for p_ind = p_cent_ind - no_pixels_considered:p_cent_ind+no_pixels_considered
                        if p_ind >= 1 && p_ind <= p_num
                            cdf_start = p_arr(p_ind)-p_res/2;
                            cdf_end = p_arr(p_ind)+p_res/2;
                            cdf1 = (1/2)*(1+erf((cdf_start-p_val)/(sigma_det*sqrt(2))));
                            cdf2 = (1/2)*(1+erf((cdf_end-p_val)/(sigma_det*sqrt(2))));
                            ratio = cdf2 - cdf1;
                            sinogram(p_ind,theta_ind) = sinogram(p_ind,theta_ind) + ratio*mag*scaling_factor;
                        end
                    end
                end
 
            end
        end
    end
%     disp(['Progress: ', num2str(theta_ind), ' / ', num2str(phi_num)]);
end
 
end