function imgheaders = loadDicomImageInfo(imagedir)

imagefiles = dir([imagedir filesep '*']);
imagefiles = imagefiles(~[imagefiles.isdir]);

imgheaders = cell(1, length(imagefiles));

sliceno = NaN(1,length(imagefiles));
for i  = 1:length(imagefiles)
    
    info = dicominfo(fullfile(imagedir, imagefiles(i).name));
    sliceno(i) = info.InstanceNumber;
    imgheaders{info.InstanceNumber} = info;
    
end

imgheaders = imgheaders(min(sliceno):max(sliceno));

end
