function TBR = generate_TBR(tumor_bg_intensity)
% Generate tumor to background ratio with kernel distribution

pd = fitdist(tumor_bg_intensity,'kernel');
TBR = random(pd);

end