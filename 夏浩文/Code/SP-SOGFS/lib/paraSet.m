function [dc, rho] = paraSet(dist, percNeigh, kernel)
    
    distRow = squareform(dist, 'tovector'); % 变为距离横向量
    sortDistRow = sort(distRow);
    [NE, ~] = size(dist); % NE: num 
    dc = sortDistRow(round((NE*(NE-1)/2)*percNeigh));
    
    if strcmp(kernel, 'Gauss')
        rho = sum(exp(-(dist/dc).^2) , 1) - exp(-(0/dc).^2); % local density estimation using Guass Kernel /dist 含有 0
    elseif strcmp(kernel, 'Cut-off')
        rho = sum(dist < dc, 1) - 1; % the number of points that are closer than dc to point i
    else
        % NO NOTHING, just for further work ...
    end
    
end