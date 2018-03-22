% find colormap which colors the distance from zero - zero is always white
% and the larger distance from zero, the darker color, up to black

function ColMap = MyCoolMapRed(bottom, top)

%% checks
if bottom>=top
    error('bottom must be smaller than top')
end

%% objects
range  = (top - bottom);
ncols  = 100*round(range/30) + 1;
if ncols  == 1
    ncols  = 101;
end

%% Variant check
if and(bottom <0, top<=0)
    Variant = 1;
end
if and(bottom >=0, top>0)
    Variant = 2;
end
if and(bottom <0, top>0)
    Variant = 3;
end

switch Variant
    %-------------------------------------------------------------- 
    case 1 % bottom and top ''less'' than zero - red color
        if top == 0
            ColMap   = ((0:(ncols-1))/(ncols-1))';
            ColMap   = repmat(ColMap, 1, 2);
            ColMap   = [ColMap, ones(ncols,1)];
        else
            ColMap   = 0.9*((0:(ncols-1))/(ncols-1) )';
            ColMap   = repmat(ColMap, 1, 2);
            ColMap   = [ColMap, ones(ncols,1)];
        end
    %--------------------------------------------------------------    
    case 2 % bottom and top ''greater'' than zero
        if bottom ==0
            ColMap   = fliplr((0:(ncols-1))/(ncols-1) )';
            ColMap   = repmat(ColMap, 1, 2);
            ColMap   = [ones(ncols,1), ColMap];
        else
            ColMap   = 0.9*(fliplr((0:(ncols-1))/(ncols-1) ))';
            ColMap   = repmat(ColMap, 1, 2);
            ColMap   = [ones(ncols,1), ColMap];
        end
    %--------------------------------------------------------------     
    case 3 % bottom ''less'' than zero, top ''greater'' than zero
        distt    = top - bottom;
        posperc  = abs(bottom)/distt;
        poss     = round(posperc*ncols);
        if poss > ncols/2
            ColMap1   = (0:(poss-1))'/(poss-1);
            ColMap1   = repmat(ColMap1, 1, 2);
            ColMap1   = [ColMap1, ones(poss,1)]; 
            ColMap2   = (fliplr(ColMap1'))';
            ColMap2   = ColMap2(1:(ncols-poss),1:2);
            ColMap2   = [ones(ncols-poss,1), ColMap2];
            ColMap    = [ColMap1;ColMap2];
        else
            ColMap2   = fliplr((0:(ncols-poss-1)))'/(ncols-poss-1);
            ColMap2   = repmat(ColMap2, 1, 2);
            ColMap2   = [ones(ncols-poss,1), ColMap2];
            ColMap1   = ColMap2(1:(poss),2:3);
            ColMap1   = fliplr((ColMap1'))';
            ColMap1   = [ColMap1, ones(poss,1)];
            ColMap    = [ColMap1; ColMap2];
        end
     %-------------------------------------------------------------- 
end

end