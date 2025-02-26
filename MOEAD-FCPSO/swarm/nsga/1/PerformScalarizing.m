
function params = PerformScalarizing(z, params)

    nObj = size(z,1);
    nPop = size(z,2);
    
    if ~isempty(params.smin)
        zmax = params.zmax;
        smin = params.smin;
    else
        zmax = zeros(nObj, nObj);
        smin = inf(1,nObj);
    end
    
    parfor j = 1:nObj
       
        w = GetScalarizingVector(nObj, j);
        
        s = zeros(1,nPop);
        for i = 1:nPop
            s(i) = max(z(:,i)./w);  %z��ת�����Ŀ�꺯��ֵ
        end

        [sminj, ind] = min(s);
        
        if sminj < smin(j)
            zmax(:, j) = z(:, ind);  %�ڽؾ��ϵĵ�
            smin(j) = sminj;  %�ؾ�
        end
        
    end
    
    params.zmax = zmax;
    params.smin = smin;
    
end

function w = GetScalarizingVector(nObj, j)

    epsilon = 1e-6;
    
    w = epsilon*ones(nObj, 1);   %����3������w����(1 0 0 ) 0 1 0 0 0 1
    
    w(j) = 1;

end