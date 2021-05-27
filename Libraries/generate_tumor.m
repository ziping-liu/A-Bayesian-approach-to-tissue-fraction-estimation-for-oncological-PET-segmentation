function [output, lesion_mask, checkpoint] = generate_tumor(tumor_shape_data, tumor_diameter, max_iter, iter_ind, loc_ind, choice_ind)
% Generates random tumors of different shape, size, intensity and
% heterogenity based on patient data

% Inputs:
% tumor_shape_data - 4x5xPatients (using 5 harmonic coefficients)
% tumor_size - patient tumor diameter data
% tumor_intensity, tumor_variance - patient tumor intensity/variance data
% heterogenous - boolean, if true then generate heterogenity
% lower intensity in the center of the lesion
% max_iter - maximum allowed iterations to attain a resonable tumor shape

% Outputs:
% output -
% lesion_mask -  binary mask of tumor

% Generate random tumor shape

checkpoint = 0;

tumor_diameter = 32.*tumor_diameter;
% tumor_volume = 32^2.*tumor_volume;

counter = 0;
while counter < max_iter % stop generating tumors after max_iter or after you get a valid tumor shape
    counter = counter + 1;
    rFSDs = zeros(4,5); % generated Efourier coefficients
    for x = 1:4
        for y = 1:5
            tmp_seq = squeeze(tumor_shape_data(x,y,:));
            pd = fitdist(tmp_seq,'kernel');
            if all(tmp_seq==tmp_seq(1))
                rFSDs(x,y) = tmp_seq(1);
            else
                % Kernel dist
                rFSDs(x,y) = random(pd);
            end
        end
    end
    outln = rEfourier(rFSDs, size(rFSDs,2), 5000);
    outln = [outln; outln(1,:)]; % to connect the head and tail
    outln = abs(flipud(outln.*50)); % increase resolution
    
    bw = poly2mask(outln(:,2), outln(:,1), 41*50, 31*50);

    if sum(bw(:))==0
        disp('bw is zero');
    end

    % Check for self intersection
    [x0,~,~] = selfintersect(outln(:,2),outln(:,1));
    if isempty(x0)
        [I,J] = ind2sub(size(bw),find(bw>0));
        output = bw(min(I):max(I), min(J):max(J));
        break;
    end
end

if counter == max_iter
    checkpoint = 1;
    disp(['ERROR: Too many iterations in ', num2str(iter_ind), 'th lung slice and ', num2str(loc_ind),'th tumor position and ',num2str(choice_ind),'th choice']);
    if ~exist('output')
        output = zeros(100*8);
    end
end

% Create lesion mask
lesion_mask = output > 0;

% Model heterogenity in tumor shape
% if heterogenous == 1
%     output = generate_tumor_heterogeneity(output, 0.2, 0.8);
% end

% Apply random rotation
angle = unifrnd(0,360);
output = imrotate(output, angle);
lesion_mask = imrotate(lesion_mask, angle);
[row,col] = ind2sub(size(lesion_mask),find(lesion_mask>0));
output = output(min(row):max(row), min(col):max(col));
lesion_mask = lesion_mask(min(row):max(row), min(col):max(col));
% fprintf('Angle of rotation: %.2f\n',angle);

% Model tumor size keeping the tumor volume within the bounds of the real
% patient data
diameter = generate_tumor_diameter(tumor_diameter);
[x,y] = size(output);
scaling_factor = sqrt(diameter^2/(x^2+y^2));
x = ceil(scaling_factor*x);
y = ceil(scaling_factor*y);
lesion_mask_resize = imresize(lesion_mask, [x,y]);
while (sum(lesion_mask_resize(:)) < (1024*5) || sum(lesion_mask_resize(:)) > (1024*1024))

%     sum(lesion_mask_resize(:)) < (1024*5)

    diameter = generate_tumor_diameter(tumor_diameter);
    [x,y] = size(output);
    scaling_factor = sqrt(diameter^2/(x^2+y^2));
    x = ceil(scaling_factor*x);
    y = ceil(scaling_factor*y);
    lesion_mask_resize = imresize(lesion_mask, [x,y]);
end
lesion_mask = lesion_mask_resize;
output = imresize(output, [x,y]);
output = round(output);
output = output .* lesion_mask;

% Remove heterogenity if lesion mask is too small
% if sum(lesion_mask(:)) <= 1024*100
%     output(output==2) = 1;
% %     disp('Heterogeneity removed');
% end

% Check if there are multiple objects
cc = bwconncomp(lesion_mask);
if cc.NumObjects ~= 1
    checkpoint = 1;
    disp(['Multiple ojects in tumor slice in ', num2str(iter_ind), 'th lung slice and ', num2str(loc_ind),'th tumor position and ',num2str(choice_ind),'th choice']);
end

end
