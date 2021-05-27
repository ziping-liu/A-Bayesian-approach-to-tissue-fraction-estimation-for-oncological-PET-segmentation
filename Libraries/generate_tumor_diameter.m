function diameter = generate_tumor_diameter(tumor_diameter)
% Generate tumor diameter with kernel distribution

pd = fitdist(tumor_diameter,'kernel');
diameter = random(pd);

if diameter < min(tumor_diameter)
    diameter = generate_tumor_diameter(tumor_diameter);
end

end
