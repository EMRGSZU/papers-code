function sp=CreateSubProblems(nObj,nPop,T)
% ��������������
    empty_sp.lambda=[];
    empty_sp.Neighbors=[];

    sp=repmat(empty_sp,nPop,1);
    
    %theta=linspace(0,pi/2,nPop);
     [W,~] = UniformPoint(153,3);
    % W=W((1:end-3),:);
%      W=W((4:end),:);
    % W=1./W;
    W=W((4:end),:);
    for i=1:nPop
        sp(i).lambda=W(i,:);          %spΪ��
        
        %sp(i).lambda=[cos(theta(i))
        %              sin(theta(i))];
    end

    LAMBDA=[sp.lambda];
    LAMBDA=reshape(LAMBDA,3,150)';
    D=pdist2(LAMBDA,LAMBDA);

    for i=1:nPop
        [~, SO]=sort(D(i,:));
        sp(i).Neighbors=SO(1:T);
    end

end