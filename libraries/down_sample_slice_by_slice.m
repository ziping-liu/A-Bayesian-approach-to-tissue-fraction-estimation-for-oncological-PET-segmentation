function [TFA_map] = down_sample_slice_by_slice(object,object_size,block_size)
    
    recon_size = object_size/block_size;
    
    if rem(object_size,recon_size) ~= 0
       error('object size divided by recon size is not an integer.');  
    end
    
    axial_slices = size(object,3);
    
    TFA_map = zeros(recon_size,recon_size,axial_slices);
    
    for dim_1 = 1:block_size:object_size
        for dim_2 = 1:block_size:object_size
            for dim_3 = 1:axial_slices 
                ssum = 0;
                for sub_dim_1 = 0:(block_size-1)   
                    for sub_dim_2 = 0:(block_size-1)        
                        ssum = ssum + object(dim_1+sub_dim_1, dim_2+sub_dim_2, dim_3);     
                    end  
                end
                TFA_map(ceil(dim_1/block_size), ceil(dim_2/block_size), dim_3) = ssum/(block_size^2);
            end 
        end
    end
end