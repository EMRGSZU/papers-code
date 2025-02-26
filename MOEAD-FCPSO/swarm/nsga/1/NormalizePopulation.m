 %超级辣鸡的标准化pop
function [pop, params] = NormalizePopulation(pop, params)

    params.zmin = UpdateIdealPoint(pop, params.zmin);  %返回最小的理想点
    
    fp = [pop.Cost] - repmat(params.zmin, 1, numel(pop)); %转换目标函数，全部减去最小值
    
    params = PerformScalarizing(fp, params);
    
    a = FindHyperplaneIntercepts(params.zmax); %nsga3算法缺陷在混乱情况下的
    %会出现某个截距不正常，小于等于理想最优值，这是我们取该目标的最大值作为截距。
    %对无法构建超平面的情况进行处理（超级辣鸡处理方式，需要后续优化）
    cind=find(a==0);
    nind=find(isnan(a));
    if size(cind,1)~=0
    [dvalue,~]=max(fp,[],2);
    a(cind)=dvalue(cind);
    end
    if size(nind,1)~=0
      [dvalue,~]=max(fp,[],2);
      a(nind)=dvalue(nind);
    end
    
     ccind=find(a==0);
      nnind=find(isnan(a));
    if size(ccind,1)~=0
        a(ccind)=1e-3;
    end
    if  size(nnind,1)~=0
      a(nnind)=1e-3;
    end
    
    parfor i = 1:numel(pop)
        pop(i).NormalizedCost = fp(:,i)./a;
%         if size(find(isnan(pop(i).NormalizedCost)),1)>=1
%             disp('laji');
%         end
    end
    
end

function a = FindHyperplaneIntercepts(zmax)

    w = ones(1, size(zmax,2))/zmax;
    
    a = (1./w)';

end