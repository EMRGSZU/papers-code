 %���������ı�׼��pop
function [pop, params] = NormalizePopulation(pop, params)

    params.zmin = UpdateIdealPoint(pop, params.zmin);  %������С�������
    
    fp = [pop.Cost] - repmat(params.zmin, 1, numel(pop)); %ת��Ŀ�꺯����ȫ����ȥ��Сֵ
    
    params = PerformScalarizing(fp, params);
    
    a = FindHyperplaneIntercepts(params.zmax); %nsga3�㷨ȱ���ڻ�������µ�
    %�����ĳ���ؾ಻������С�ڵ�����������ֵ����������ȡ��Ŀ������ֵ��Ϊ�ؾࡣ
    %���޷�������ƽ���������д���������������ʽ����Ҫ�����Ż���
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