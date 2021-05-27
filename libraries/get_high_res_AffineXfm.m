function A = get_high_res_AffineXfm(headers,block_size)

% Modified from https://github.com/ulrikls/dicomrt2matlab

N = length(headers);

dr = headers{1}.PixelSpacing(1)/block_size;
dc = headers{1}.PixelSpacing(2)/block_size;

F(:,1) = headers{1}.ImageOrientationPatient(1:3);
F(:,2) = headers{1}.ImageOrientationPatient(4:6);

T1 = headers{1}.ImagePositionPatient;
TN = headers{end}.ImagePositionPatient;

k = (T1 - TN) ./ (1 - N);

A = [[F(1,1)*dr F(1,2)*dc k(1) T1(1)]; ...
     [F(2,1)*dr F(2,2)*dc k(2) T1(2)]; ...
     [F(3,1)*dr F(3,2)*dc k(3) T1(3)]; ...
     [0         0         0    1    ]];
 
end

