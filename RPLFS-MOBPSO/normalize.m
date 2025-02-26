function [normalized_cost, params] = normalize(pop, params)
% 归一化
    % 更新理想点，zmin就是ideal point
    params.zmin = min([params.zmin, [pop.Cost]], [], 2);

    fp = [pop.Cost] - repmat(params.zmin, 1, numel(pop));

    [params.zmax, params.smin] = scalarize(fp, params.zmax, params.smin);

    a = find_hyperplane_intercepts(params.zmax);

    normalized_cost = fp ./ repmat(a, 1, size(fp, 2));

end

function a = find_hyperplane_intercepts(zmax)

    % disp('zmax');
    % disp(zmax);
    w = ones(1, size(zmax, 2)) / (zmax+1e-6*diag(diag(ones(size(zmax, 2)))));
    a = (1 ./ w)';

end
