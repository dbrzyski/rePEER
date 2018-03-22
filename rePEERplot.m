function rePEERplot(out, type)

%% Nargin
if nargin ==1
    type = 'b';
end

%% Switch case
switch type
	case 'b'
        estimate  = out.b;
        CI        = out.CIb;
        CIsigFind = find(CI(:,1).*CI(:,2)>0);
        plotTitle = 'rePEER prediction of b with analytically derived CI';
	case 'beta'
        estimate = out.beta;
        CI        = out.CIbeta;
        CIsigFind = find(CI(:,1).*CI(:,2)>0);
        plotTitle = 'rePEER estimate of beta with analytically derived CI';
	case 'bBS'
       if isfield(out, 'CIbBS')
            estimate  = out.b;
            CI        = out.CIbBS;
            CIsigFind = find(CI(:,1).*CI(:,2)>0);
            plotTitle = 'rePEER prediction of b with bootstrap derived CI';
       else
            error('The field corresponding to bootstrap CI was not found')
       end
	case 'betaBS'
      if isfield(out, 'CIbetaBS')
            estimate  = out.beta;
            CI        = out.CIbetaBS;
            CIsigFind = find(CI(:,1).*CI(:,2)>0);
            plotTitle = 'rePEER estimate of beta with bootstrap derived CI';
      else
            error('The field corresponding to bootstrap CI was not found')
      end
	otherwise
        error('The second argument should be "b", "beta", "bBS" or "betaBS"')
end

%% Objects
p   = size(CI,1);
if p ==1
    x   = [0.9; 1; 1.1];
    x2  = [x; flipud(x)];
    CI  = [CI; CI; CI];
else
    x   = (1:p)';
    x2  = [x; flipud(x)];
end
fillcol = [.91 .91 .91];
limm    = max(max(abs(CI)));
maxLimm = max(max(CI));
minLimm = min(min(CI));

%% Plot
f = figure('name','rePEER estimate');
plot(estimate,'.','color', 'blue', 'MarkerSize', 12)
hold on
inBetween  = [CI(:,1); flipud(CI(:,2))];
h          = fill(x2, inBetween, fillcol, 'edgecolor', 'none');
set(h,'facealpha',.5)
line(x, CI(:,1),'color','k','linewidth',.1, 'LineStyle',':')
line(x, CI(:,2),'color','k','linewidth',.1, 'LineStyle',':')
line(x, estimate,'color','blue','linewidth',1.1)
if ~isempty(CIsigFind)
    for vv = 1: length(CIsigFind)
        plot([CIsigFind(vv), CIsigFind(vv)], [-2*limm, 2*limm] , 'r')
    end
    %vline(CIsigFind, 'r')
end
if p==1
    xlim([0.5 1.5])
end
plot([0, 2*p], [0,0], 'black')
grid on;
title(plotTitle)
%-------------------- Plot's settings ------------------------------------
x0     = f.Position(1);
y0     = f.Position(2);
width  = 700;
height = f.Position(4);
set(gcf,'units','pixels','position',[x0, y0, width, height])
axis([0.5, p+0.5, 1.05*minLimm, 1.05*maxLimm])
end

