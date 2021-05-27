%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Description    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Ziping Liu (2020 Dec)
% Summary:
% This script includes the steps to extract PET and reader annotations from MIM software. 

% Step 1. Extract PET images from raw PET dicom files and save as mat files 
% Step 2. Generate tumor-fraction area maps based on high-resolution tumor segmentation from raw RT struct files and save as mat files
% Step 3. Prepare ground truth for DL training 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 0.

addpath(genpath('./libraries/'));

block_size = 8; % Define the resolution of high-resolution tumor mask

patient_dirname = []; % Enter the path for the current patient case
% The patient directory consists of one RT struct folder and one PET image
% folder downloaded from MIM software
export_dir_name = []; % Enter export path

patient_dir = dir(patient_dirname);

dirFlags = [patient_dir.isdir] & ~strcmp({patient_dir.name},'.') & ~strcmp({patient_dir.name},'..');
patient_subFolders = patient_dir(dirFlags);

C = {patient_subFolders.name};
PT_idx = ~cellfun('isempty',regexp({patient_subFolders.name},'Saved.PT_','match'));
RTstrct_idx = ~cellfun('isempty',regexp({patient_subFolders.name},'_RTst_','match')); 

PT_dir = [patient_dirname,'/',C{PT_idx}];
imageheaders = loadDicomImageInfo(PT_dir);
    
RTstruct_dir = [patient_dirname,'/',C{RTstrct_idx}];
RTstruct_file = dir(fullfile(RTstruct_dir, '*.dcm'));
RTstruct_filename = [RTstruct_dir,'/',RTstruct_file.name];
rtssheader = dicominfo(RTstruct_filename);

%% Step 1.
PT_files = dir(fullfile(PT_dir, '*.dcm'));
PT_filenames = cellfun(@(x)fullfile(PT_dir, x), {PT_files.name}, 'uni', 0);
warning('off')
PT_axial_infos = cellfun(@dicominfo, PT_filenames);
[~, inds] = sort([PT_axial_infos.SliceLocation],'descend');
PT_axial_infos = PT_axial_infos(inds);

PT_rescale_slope = PT_axial_infos(1).RescaleSlope;
PT_rescale_intercept = PT_axial_infos(1).RescaleIntercept;

recon = zeros(length(PT_axial_infos),PT_axial_infos(1).Width,PT_axial_infos(1).Height);

for PT_ind = 1:length(PT_axial_infos)
    PT_slice = dicomread(PT_axial_infos(PT_ind));
    recon(PT_ind,:,:) = double(PT_slice).*PT_rescale_slope + PT_rescale_intercept;
end

save([export_dir_name,'\recon.mat'],'recon');

%% Step 2-3.

high_res_contours = generate_high_res_contour(rtssheader, imageheaders, block_size);

contours2cell = squeeze(struct2cell(high_res_contours));
ROI_names = contours2cell(1,:);
primary_tumor_index = find(strcmp(ROI_names,'ROI-1'));
high_res_primary_tumor_mask = double(permute(high_res_contours(primary_tumor_index).Segmentation,[2 1 3]));

primary_TFA_map = down_sample_slice_by_slice(high_res_primary_tumor_mask,size(high_res_primary_tumor_mask,1),block_size);

true = zeros(length(PT_axial_infos),PT_axial_infos(1).Width,PT_axial_infos(1).Height,2);
% Groud truth is of size Num_axial_slices x 128 x 128 x 2, where the first channel is tumor class and the second channel is background
true(:,:,:,1) = primary_TFA_map;
true(:,:,:,2) = 1 - primary_TFA_map;

save([export_dir_name,'\true.mat'],'true');