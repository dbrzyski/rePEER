function MyHeatmapRed(b, tit, bottom, top)
if nargin == 1
    tit ="";
end

if nargin <= 2
    b       = double(b);
    bottom  = min(min(b));
    top     = max(max(b));
end

%% heatmap
margins = [0.05, 0.00001];
f1 = figure('name','Estimates');
subplot_tight(1,1,1,margins );
imagesc(b);
colormap(MyCoolMapRed(bottom ,top));
title(tit);
axis equal;
axis tight;
caxis manual
caxis([bottom  top]);
colorbar
% 
% % Plot's settings
% x0     = 100;
% y0     = 100;
% width  = 600;
% height = 500;
% set(gcf, 'units', 'points', 'position', [x0,y0,width,height])


end
