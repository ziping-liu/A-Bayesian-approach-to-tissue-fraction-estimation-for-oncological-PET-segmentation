function [centerX,centerY] = generate_lump_center(min_lumpy_row,max_lumpy_row,min_lumpy_col,max_lumpy_col,lumpy_template)
% Generate tumor diameter with kernel distribution

tumor_row_ind = min_lumpy_row + randperm(max_lumpy_row-min_lumpy_row,1);
tumor_col_ind = min_lumpy_col + randperm(max_lumpy_col-min_lumpy_col,1);

centerX = tumor_col_ind;
centerY = tumor_row_ind;

if lumpy_template(tumor_row_ind,tumor_col_ind) == 0
    [centerX,centerY] = generate_lump_center(min_lumpy_row,max_lumpy_row,min_lumpy_col,max_lumpy_col,lumpy_template);
end

end
